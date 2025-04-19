#include <time.h>
#include <vector>
#include <chrono>
#include <boost/optional.hpp>
#include <libcamera/stream.h>
#include "core/frame_info.hpp"
#include "core/rpicam_app.hpp"
#include "post_processing_stages/post_processing_stage.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using Stream = libcamera::Stream;

#define LOG_DEBUG(stage, msg) std::cerr << "[" << stage << "] " << msg << std::endl

struct OverlayConfig 
{
    std::string text;
    int fg;
    int bg;
    bool has_bg;     
    double scale;
    int thickness;
    double alpha;
    std::string x_str;
    std::string y_str;
    int x;
    int y;
    uint64_t update_interval;
    bool is_dynamic;
    int border_width;  
    int border_color; 
    bool has_border;   

    // Cached ROI data
    cv::Mat cached_roi;
    int cached_x;
    int cached_y;
    int cached_width;
    int cached_height;
    uint64_t last_update;
};

class OverlayCVStage : public PostProcessingStage {
public:
    OverlayCVStage(RPiCamApp *app) : PostProcessingStage(app) {}
    char const *Name() const override { return "overlay_cv"; }
    void Read(boost::property_tree::ptree const &params) override;
    void Configure() override;
    bool Process(CompletedRequestPtr &completed_request) override;

private:
    Stream *stream_;
    StreamInfo info_;
    std::vector<OverlayConfig> overlays_;
    void ParsePosition(const std::string &s, int &pos, int base);
    std::string FormatText(const std::string &text, const FrameInfo &info);
    void GenerateCache(OverlayConfig &config, const std::string &text);
};

void OverlayCVStage::Read(boost::property_tree::ptree const &params)
{
    LOG_DEBUG(Name(), "Read: got array of overlays:");
    std::stringstream ss;
    write_json(ss, params);
    LOG_DEBUG(Name(), ss.str());

    for (auto const &item : params)
    {
        auto const &overlay = item.second;  

        OverlayConfig config;
        config.text            = overlay.get<std::string>("text");
        config.fg              = overlay.get<int>("fg", 255);
        
        if (overlay.find("bg") != overlay.not_found())
        {
            config.bg = overlay.get<int>("bg");
            config.has_bg = true;
        }
        else
        {
            config.bg = 0;
            config.has_bg = false;
        }
        
        config.scale           = overlay.get<double>("scale", 1.0);
        config.thickness       = overlay.get<int>("thickness", 2);
        config.alpha           = overlay.get<double>("alpha", 0.5);
        config.x_str           = overlay.get<std::string>("x", "0");
        config.y_str           = overlay.get<std::string>("y", "0");
        config.update_interval = overlay.get<int>("update_interval", 1000);
        config.is_dynamic      = (config.text.find('%') != std::string::npos);
        
        if (overlay.find("border_width") != overlay.not_found())
        {
            config.border_width = overlay.get<int>("border_width");
            config.border_color = overlay.get<int>("border_color", 0);
            config.has_border = true;
        }
        else
        {
            config.border_width = 0;
            config.has_border = false;
        }

        config.cached_roi = cv::Mat();
        config.last_update = 0;

        overlays_.push_back(config);
    }
}

void OverlayCVStage::Configure() {
    stream_ = app_->GetMainStream();
    if (!stream_ || stream_->configuration().pixelFormat != libcamera::formats::YUV420)
        throw std::runtime_error("only YUV420 supported");
    info_ = app_->GetStreamInfo(stream_);
    for (auto &c : overlays_) {
        ParsePosition(c.x_str, c.x, info_.width);
        ParsePosition(c.y_str, c.y, info_.height);
        c.scale     = c.scale * info_.width / 1200;
        c.thickness = std::max<int>(c.thickness * info_.width / 700, 1);
    }
}

std::string OverlayCVStage::FormatText(const std::string &t, const FrameInfo &info) {
    auto s = info.ToString(t);
    char buf[256]; time_t tt=time(NULL); tm *tm_p=localtime(&tt);
    return (strftime(buf, sizeof(buf), s.c_str(), tm_p) !=0) ? std::string(buf) : s;
}

void OverlayCVStage::GenerateCache(OverlayConfig &config, const std::string &text) {
    if (text.empty()) {
        config.cached_roi.release();
        return;
    }

    int font = FONT_HERSHEY_SIMPLEX;
    int baseline = 0;
	
    Size text_size = getTextSize(text, font, config.scale, config.thickness, &baseline);

    int x_pos = std::max(0, std::min(config.x, static_cast<int>(info_.width - text_size.width)));
    int y_pos = std::max(text_size.height + baseline, std::min(config.y, static_cast<int>(info_.height - baseline)));

    Rect bg_rect(x_pos, y_pos - text_size.height, text_size.width, text_size.height + baseline);
    cv::Mat cached_roi(bg_rect.height, bg_rect.width, CV_8UC1);

    if (config.has_bg)
	{
        cached_roi.setTo(config.bg);
    } else {
        cached_roi.setTo(0); 
    }

    if (config.has_border) {
        rectangle(cached_roi, Rect(0, 0, bg_rect.width, bg_rect.height),
                  config.border_color, config.border_width);
    }

    putText(cached_roi, text, Point(0, text_size.height), font, config.scale,
            config.fg, config.thickness, LINE_AA);

    config.cached_x = bg_rect.x;
    config.cached_y = bg_rect.y;
    config.cached_width = bg_rect.width;
    config.cached_height = bg_rect.height;
    config.cached_roi = cached_roi.clone();
}

bool OverlayCVStage::Process(CompletedRequestPtr &completed_request)
{
    BufferWriteSync w(app_, completed_request->buffers[stream_]);
    libcamera::Span<uint8_t> buffer = w.Get()[0];
    FrameInfo info(completed_request);
    
    uint8_t *ptr = (uint8_t *)buffer.data();
    Mat frame(info_.height, info_.width, CV_8U, ptr, info_.stride);

    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    for (auto &config : overlays_)
    {
        std::string text = config.is_dynamic ? FormatText(config.text, info) : config.text;
        if (text.empty()) continue;

        if (config.is_dynamic) {
            if (current_time - config.last_update >= config.update_interval) {
                GenerateCache(config, text);
                config.last_update = current_time;
            }
        } else {
            if (config.cached_roi.empty()) {
                GenerateCache(config, text);
            }
        }

        if (!config.cached_roi.empty()) {
            Rect roi_rect(config.cached_x, config.cached_y, config.cached_width, config.cached_height);
            if (roi_rect.x >= 0 && roi_rect.y >= 0 &&
                roi_rect.x + roi_rect.width <= frame.cols &&
                roi_rect.y + roi_rect.height <= frame.rows) {
                Mat dst_roi = frame(roi_rect);
                config.cached_roi.copyTo(dst_roi);
            }
        }
    }

    return false;
}

static PostProcessingStage* CreateA(RPiCamApp*app) { return new OverlayCVStage(app); }
static RegisterStage regA("overlay_cv", &CreateA);