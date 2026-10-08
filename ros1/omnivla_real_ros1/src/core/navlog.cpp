#include "omnivla/navlog.hpp"

#include <sys/stat.h>

#include <cmath>
#include <ctime>
#include <fstream>
#include <stdexcept>

namespace omnivla {

namespace {

constexpr int kNumWp = 8;
constexpr int kWaypointIndex = 4;  // pred_bearing に使う点 (navlog.py と同じ)

void mkdirs(const std::string& path) {
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    cur += path[i];
    if ((path[i] == '/' && i > 0) || i + 1 == path.size()) {
      ::mkdir(cur.c_str(), 0777);
      ::chmod(cur.c_str(), 0777);
    }
  }
}

bool exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

std::string now_str(const char* fmt) {
  char buf[64];
  std::time_t t = std::time(nullptr);
  std::strftime(buf, sizeof(buf), fmt, std::localtime(&t));
  return buf;
}

void write_json(const std::string& path, const nlohmann::json& j) {
  std::ofstream f(path);
  f << j.dump(2);
  ::chmod(path.c_str(), 0666);
}

void write_bytes(const std::string& path, const Bytes& b) {
  if (!b) return;
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(b->data()), static_cast<std::streamsize>(b->size()));
}

// navlog.py の _fmt: None -> "", float -> "%.5g", nan -> ""
std::string f5(double v) {
  if (std::isnan(v)) return "";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.5g", v);
  return buf;
}
std::string f5(const std::optional<double>& v) { return v ? f5(*v) : ""; }

double deg(double r) { return r * 180.0 / 3.14159265358979323846; }

}  // namespace

NavLogger::NavLogger(const std::string& root_in, nlohmann::json meta, const std::vector<FramePtr>& goals, bool save_raw)
    : save_raw_(save_raw) {
  std::string root = root_in;
  while (root.size() > 1 && root.back() == '/') root.pop_back();
  mkdirs(root);
  const std::string stamp = now_str("%Y%m%d_%H%M%S");
  dir_ = root + "/" + stamp;
  for (int k = 1; exists(dir_); ++k) dir_ = root + "/" + stamp + "_" + std::to_string(k);
  mkdirs(dir_);
  mkdirs(dir_ + "/goals");
  if (save_raw_) mkdirs(dir_ + "/raw");
  meta["started_at"] = now_str("%Y-%m-%d %H:%M:%S");
  if (meta.contains("final_goal_pose") && meta["final_goal_pose"].is_array()) {
    const auto& g = meta["final_goal_pose"];
    final_goal_ = Pose{g[0].get<double>(), g[1].get<double>(), g[2].get<double>()};
  }
  write_json(dir_ + "/meta.json", meta);
  for (size_t i = 0; i < goals.size(); ++i) write_bytes(dir_ + "/goals/" + std::to_string(i) + ".jpg", goals[i]->data);
  const std::string csv = dir_ + "/steps.csv";
  csv_ = std::fopen(csv.c_str(), "w");
  if (!csv_) throw std::runtime_error("cannot write " + csv);
  ::chmod(csv.c_str(), 0666);
  std::fputs("step,sim_time,x,y,yaw,subgoal,num_nodes,sg_x,sg_y,sg_yaw,goal_local_x,goal_local_y,goal_bearing_deg,"
             "pred_bearing_deg,modality,controller,v,w,latency,dist,similarity,state",
             csv_);
  for (int i = 0; i < kNumWp; ++i) std::fprintf(csv_, ",wp%d_x,wp%d_y,wp%d_yaw_deg", i, i, i);
  std::fputs("\r\n", csv_);  // Python の csv.writer と同じ改行
  std::fflush(csv_);
}

NavLogger::~NavLogger() {
  if (!closed_) close("shutdown");
}

void NavLogger::event(std::optional<double> t, const std::string& text) {
  std::lock_guard<std::mutex> lk(mu_);
  if (closed_) return;
  std::ofstream f(dir_ + "/events.log", std::ios::app);
  char buf[32];
  if (t)
    std::snprintf(buf, sizeof(buf), "%.2f", *t);
  else
    std::snprintf(buf, sizeof(buf), "-");
  f << "[" << buf << "] " << text << "\n";
}

void NavLogger::track_pose(double t, const std::optional<Pose>& pose) {
  if (!pose) return;
  if (!first_pose_) {
    first_pose_ = pose;
    first_time_ = t;
  }
  if (last_pose_) path_len_ += std::hypot(pose->x - last_pose_->x, pose->y - last_pose_->y);
  last_pose_ = pose;
  last_time_ = t;
  if (final_goal_)
    min_final_dist_ = std::min(min_final_dist_, std::hypot(pose->x - final_goal_->x, pose->y - final_goal_->y));
}

void NavLogger::step(int step, double t, const std::optional<Pose>& pose, int subgoal, int num_nodes,
                     const std::optional<Pose>& sg, const std::optional<std::pair<double, double>>& gl, int modality,
                     const std::string& controller, double v, double w, double latency, const Waypoints& wps,
                     std::optional<double> dist, std::optional<double> similarity, const std::string& state,
                     const FramePtr& raw) {
  std::lock_guard<std::mutex> lk(mu_);
  if (closed_) return;
  ++n_steps_;
  track_pose(t, pose);
  const int k = std::min<int>(kWaypointIndex, static_cast<int>(wps.size()) - 1);
  const double pred_bearing = deg(std::atan2(wps[k][1], wps[k][0]));
  std::string row = std::to_string(step) + "," + f5(t) + ",";
  row += pose ? f5(pose->x) + "," + f5(pose->y) + "," + f5(pose->yaw) : std::string(",,");
  row += "," + std::to_string(subgoal) + "," + std::to_string(num_nodes) + ",";
  row += sg ? f5(sg->x) + "," + f5(sg->y) + "," + f5(sg->yaw) : std::string(",,");
  row += ",";
  row += gl ? f5(gl->first) + "," + f5(gl->second) + "," + f5(deg(std::atan2(gl->second, gl->first))) : std::string(",,");
  row += "," + f5(pred_bearing) + "," + std::to_string(modality) + "," + controller + "," + f5(v) + "," + f5(w) + "," +
         f5(latency) + "," + f5(dist) + "," + f5(similarity) + "," + state;
  for (int i = 0; i < kNumWp && i < static_cast<int>(wps.size()); ++i)
    row += "," + f5(wps[i][0]) + "," + f5(wps[i][1]) + "," + f5(deg(std::atan2(wps[i][3], wps[i][2])));
  row += "\r\n";
  std::fputs(row.c_str(), csv_);
  std::fflush(csv_);
  if (save_raw_ && raw && raw->data) {
    char name[32];
    std::snprintf(name, sizeof(name), "/raw/%06d.%s", step, raw->is_jpeg() ? "jpg" : (raw->kind == "encoded" ? "png" : "bin"));
    write_bytes(dir_ + name, raw->data);
  }
}

nlohmann::json NavLogger::close(const std::string& reason, bool reached) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) return nlohmann::json::object();
  }
  event(last_time_, "run closed: " + reason);
  std::lock_guard<std::mutex> lk(mu_);
  closed_ = true;
  if (csv_) std::fclose(csv_);
  csv_ = nullptr;
  auto pose_json = [](const std::optional<Pose>& p) {
    return p ? nlohmann::json::array({p->x, p->y, p->yaw}) : nlohmann::json(nullptr);
  };
  nlohmann::json s = {{"reason", reason},
                      {"reached", reached},
                      {"steps", n_steps_},
                      {"sim_duration", first_time_ && last_time_ ? *last_time_ - *first_time_ : 0.0},
                      {"path_length_m", path_len_},
                      {"start_pose", pose_json(first_pose_)},
                      {"final_pose", pose_json(last_pose_)},
                      {"final_goal_pose", pose_json(final_goal_)},
                      {"closed_at", now_str("%Y-%m-%d %H:%M:%S")}};
  s["final_dist_to_goal"] = final_goal_ && last_pose_
                                ? nlohmann::json(std::hypot(last_pose_->x - final_goal_->x, last_pose_->y - final_goal_->y))
                                : nlohmann::json(nullptr);
  s["min_dist_to_goal"] = min_final_dist_ < 1e17 ? nlohmann::json(min_final_dist_) : nlohmann::json(nullptr);
  write_json(dir_ + "/summary.json", s);
  return s;
}

}  // namespace omnivla
