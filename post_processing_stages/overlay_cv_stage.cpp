/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2021, Raspberry Pi (Trading) Limited
 *
 * overlay_cv_stage.cpp - add multiple text overlays to image
 */

 #include <time.h>
 #include <vector>
  
 #include <libcamera/stream.h>
  
 #include "core/frame_info.hpp"
 #include "core/rpicam_app.hpp"
  
 #include "post_processing_stages/post_processing_stage.hpp"
  
 #include "opencv2/core.hpp"
 #include "opencv2/imgproc.hpp"
  
 using namespace cv;
 using Stream = libcamera::Stream;
  
 #define LOG_DEBUG(stage, msg) std::cerr << "[" << stage << "] " << msg << std::endl


 struct OverlayConfig 
 {
	 std::string text;
	 int fg;
	 int bg;
	 double scale;
	 int thickness;
	 double alpha;
	 std::string x_str;
	 std::string y_str;
	 int x;
	 int y;
	 int update_interval;
	 bool is_dynamic; 
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
  
	 void ParsePosition(const std::string &pos_str, int &pos, int base);
	 std::string FormatText(const std::string &text, const FrameInfo &info);
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
	 LOG_DEBUG(Name(), "Read: got array of overlays:");
	 std::stringstream ss;
	 write_json(ss, params);
	 LOG_DEBUG(Name(), ss.str());
 
	 for (auto const &item : params)
	 {
		 auto const &overlay = item.second;  // got to do this because params is actually all messed up because of a previous step they do to it
 
		 OverlayConfig config;
		 config.text            = overlay.get<std::string>("text");
		 config.fg              = overlay.get<int>("fg", 255);
		 config.bg              = overlay.get<int>("bg", 0);
		 config.scale           = overlay.get<double>("scale", 1.0);
		 config.thickness       = overlay.get<int>("thickness", 2);
		 config.alpha           = overlay.get<double>("alpha", 0.5);
		 config.x_str           = overlay.get<std::string>("x", "0");
		 config.y_str           = overlay.get<std::string>("y", "0");
		 config.update_interval = overlay.get<int>("update_interval", 1000);
		 config.is_dynamic      = (config.text.find('%') != std::string::npos);
 
		 overlays_.push_back(config);
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
 
		 config.scale = config.scale * info_.width / 1200;
		 config.thickness = std::max(static_cast<int>(config.thickness * info_.width / 700), 1);
	 }
 }
 
 std::string OverlayCVStage::FormatText(const std::string &text, const FrameInfo &info)
 {
	 std::string formatted_text = info.ToString(text);
	 
	 char text_with_date[256];
	 time_t t = time(NULL);
	 tm *tm_ptr = localtime(&t);
	 if (strftime(text_with_date, sizeof(text_with_date), formatted_text.c_str(), tm_ptr) != 0)
		 return std::string(text_with_date);
	 
	 return formatted_text;
 }


 bool OverlayCVStage::Process(CompletedRequestPtr &completed_request)
 {
	 BufferWriteSync w(app_, completed_request->buffers[stream_]);
	 libcamera::Span<uint8_t> buffer = w.Get()[0];
	 FrameInfo info(completed_request);
	 
	 uint8_t *ptr = (uint8_t *)buffer.data();
	 Mat frame(info_.height, info_.width, CV_8U, ptr, info_.stride);
 
	 for (auto &config : overlays_)
	 {
		 std::string text = config.is_dynamic ? FormatText(config.text, info) : config.text;
		 if (text.empty()) continue;
 
		 int font = FONT_HERSHEY_SIMPLEX;
		 int baseline = 0;
		 Size text_size = getTextSize(text, font, config.scale, config.thickness, &baseline);
 
		 int x_pos = std::max(0, std::min(config.x, static_cast<int>(info_.width) - text_size.width));
		 int y_pos = std::max(text_size.height + baseline, 
							std::min(config.y, static_cast<int>(info_.height) - baseline));
 
		 Rect bg_rect(x_pos, y_pos - text_size.height, text_size.width, text_size.height + baseline);

		 if (bg_rect.x >= 0 && bg_rect.y >= 0 && 
			 bg_rect.x + bg_rect.width <= frame.cols && 
			 bg_rect.y + bg_rect.height <= frame.rows)
		 {
			 Mat bg_roi = frame(bg_rect);
			 bg_roi = bg_roi * (1 - config.alpha) + config.bg * config.alpha;
		 }
 
		 putText(frame, text, Point(x_pos, y_pos), font, config.scale, 
				 config.fg, config.thickness, LINE_AA);
	 }
 
	 return false;
 }
 static PostProcessingStage *Create(RPiCamApp *app)
 {
	 return new OverlayCVStage(app);
 }
  
 static RegisterStage reg(NAME, &Create);