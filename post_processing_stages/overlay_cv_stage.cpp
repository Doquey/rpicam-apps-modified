/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2021, Raspberry Pi (Trading) Limited
 *
 * annotate_cv_stage.cpp - add text annotation to image
 */

#include <time.h>
#include <vector>
#include <chrono>
 
#include <libcamera/stream.h>
 
#include "core/frame_info.hpp"
#include "core/rpicam_app.hpp"
 
#include "post_processing_stages/post_processing_stage.hpp"
 
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
 
using namespace cv;
using namespace std::chrono;
using Stream = libcamera::Stream;
 
struct OverlayConfig 
{
	 std::string text;
	 int fg;
	 int bg;
	 double scale;
	 int thickness;
	 double alpha;
	 int border_width;
	 int border_color;
	 std::string x_str;
	 std::string y_str;
	 int x;
	 int y;
	 int update_interval;
	 bool is_dynamic;
	 bool has_bg;
	 bool has_border;
 };
 
struct CachedOverlay 
{
	OverlayConfig config;
	cv::Mat cache;
	steady_clock::time_point last_updated;
};
 
class OverlayCVStage : public PostProcessingStage
{
public:
	OverlayCVStage(RPiCamApp *app) : PostProcessingStage(app) {}
 
	char const *Name() const override;
 
	void Read(boost::property_tree::ptree const &params) override;
 
	void Configure() override;
 
	bool Process(CompletedRequestPtr &completed_request) override;
 
private:
	Stream *stream_;
	StreamInfo info_;
	std::vector<OverlayConfig> overlays_;
	std::vector<CachedOverlay> static_caches_;
	std::vector<CachedOverlay> dynamic_caches_;
 
	void ParsePosition(const std::string &pos_str, int &pos, int base);
	void RenderOverlay(const OverlayConfig &config, cv::Mat &im, const FrameInfo &info, bool force_update = false);
	void UpdateCachedOverlay(CachedOverlay &cached, const FrameInfo &info);
};
 
#define NAME "overlay_cv"
 
char const *OverlayCVStage::Name() const
{
	return NAME;
}
 
void OverlayCVStage::ParsePosition(const std::string &pos_str, int &pos, int base)
{
	if (pos_str.empty()) 
	{
		pos = 0;
		return;
	}
	if (pos_str.back() == '%')
	{
		double percent = std::stod(pos_str.substr(0, pos_str.size() - 1)) / 100.0;
		pos = static_cast<int>(base * percent);
	} else 
	{
		pos = std::stoi(pos_str);
	}
}
 
void OverlayCVStage::Read(boost::property_tree::ptree const &params)
{
	auto overlays_node = params.get_child_optional("annotate_cv");
	if (overlays_node)
	{
		for (auto &item : *overlays_node)
		{
			OverlayConfig config;
			boost::property_tree::ptree overlay = item.second;
			config.text = overlay.get<std::string>("text");
			config.fg = overlay.get<int>("fg", 255);
			config.bg = overlay.get<int>("bg", 0);
			config.scale = overlay.get<double>("scale", 1.0);
			config.thickness = overlay.get<int>("thickness", 2);
			config.alpha = overlay.get<double>("alpha", 0.5);
			config.border_width = overlay.get<int>("border_width", 0);
			config.border_color = overlay.get<int>("border_color", 0);
			config.x_str = overlay.get<std::string>("x", "0");
			config.y_str = overlay.get<std::string>("y", "0");
			config.update_interval = overlay.get<int>("update_interval", 1000);
			config.is_dynamic = (config.text.find('%') != std::string::npos);
			config.has_bg = overlay.find("bg") != overlay.not_found();
			config.has_border = (config.border_width > 0);
			overlays_.push_back(config);
		}
	
	}
}
 
void OverlayCVStage::Configure()
{
    stream_ = app_->GetMainStream();
    if (!stream_ || stream_->configuration().pixelFormat != libcamera::formats::YUV420)
        throw std::runtime_error("OverlayCVStage: only YUV420 format supported");
    info_ = app_->GetStreamInfo(stream_);

    for (auto &config : overlays_)
    {
        ParsePosition(config.x_str, config.x, info_.width);
        ParsePosition(config.y_str, config.y, info_.height);

        double adjusted_scale = config.scale * info_.width / 1200;
        int adjusted_thickness = std::max(static_cast<int>(config.thickness * info_.width / 700), 1);

        // Store configuration but do NOT call UpdateCachedOverlay because we need the frame info
        CachedOverlay cached;
        cached.config = config;
        cached.config.scale = adjusted_scale;
        cached.config.thickness = adjusted_thickness;

        if (!config.is_dynamic)
            static_caches_.push_back(cached);
        else
        {
            cached.last_updated = steady_clock::now();
            dynamic_caches_.push_back(cached);
        }
    }
}
void OverlayCVStage::UpdateCachedOverlay(CachedOverlay &cached, const FrameInfo &info)
{
	OverlayConfig &config = cached.config;
	std::string text = info.ToString(config.text);
 
	char dynamic_text[256];
	time_t t = time(NULL);
	tm *tm_ptr = localtime(&t);
	if (strftime(dynamic_text, sizeof(dynamic_text), text.c_str(), tm_ptr) != 0)
		text = dynamic_text;
 
	int font = FONT_HERSHEY_SIMPLEX;
	int baseline = 0;
	Size text_size = getTextSize(text, font, config.scale, config.thickness, &baseline);
 
	int border = config.border_width;
	int expanded_width = text_size.width + 2 * border;
	int expanded_height = text_size.height + baseline + 2 * border;
 
	cv::Mat overlay_mat(expanded_height, expanded_width, CV_8UC1, cv::Scalar(0));
	cv::Point text_pos(border, border + text_size.height);
 
	if (config.has_border)
	{
		cv::rectangle(overlay_mat, cv::Rect(0, 0, expanded_width, expanded_height),
					   config.border_color, FILLED);
	}
 
	if (config.has_bg) 
	{
		cv::Mat bg_roi = overlay_mat(cv::Rect(border, border, text_size.width, text_size.height + baseline));
		 bg_roi.setTo(cv::Scalar(config.bg));
	}
 
	cv::putText(overlay_mat, text, text_pos, font, config.scale,
				 config.fg, config.thickness, LINE_AA);
 
	cached.cache = overlay_mat;
	cached.last_updated = steady_clock::now();
}
bool OverlayCVStage::Process(CompletedRequestPtr &completed_request)
{
    BufferWriteSync w(app_, completed_request->buffers[stream_]);
	libcamera::Span<uint8_t> buffer = w.Get()[0];
	FrameInfo info(completed_request);
	
    uint8_t *ptr = (uint8_t *)buffer.data();
    Mat frame(info_.height, info_.width, CV_8U, ptr, info_.stride);

    // Initialize static caches on first use
    for (auto &cached : static_caches_)
    {
        if (cached.cache.empty()) // Check if cache hasn't been initialized
            UpdateCachedOverlay(cached, info);
        
        Mat roi = frame(cv::Rect(cached.config.x, cached.config.y - cached.cache.rows,
                                  cached.cache.cols, cached.cache.rows));
        cached.cache.copyTo(roi);
    }

    auto now = steady_clock::now();
    for (auto &cached : dynamic_caches_)
    {
        if (duration_cast<milliseconds>(now - cached.last_updated).count() >= cached.config.update_interval)
            UpdateCachedOverlay(cached, info);
        
        Mat roi = frame(cv::Rect(cached.config.x, cached.config.y - cached.cache.rows,
                                  cached.cache.cols, cached.cache.rows));
        cached.cache.copyTo(roi);
    }

    return false;
}
 
static PostProcessingStage *Create(RPiCamApp *app)
{
	return new OverlayCVStage(app);
}
 
static RegisterStage reg(NAME, &Create);