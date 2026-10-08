// C++ の中身を ROS 無しで動かす道具 (tests/test_cpp.py が Python の実装と突き合わせる).
//   core_check controller < cases.json      予測軌跡 -> (v, w)
//   core_check tracker    < scenario.json   サブゴールの切り替え
//   core_check replay --repo DIR --frames frames.json [--topomap DIR] [--set k=v ...] [--realtime] [--hold SEC]
//       画像列 (JPEG) とオドメトリを NavEngine に流し、推論サーバを呼んで 1 フレームごとの結果を JSON で出す.
//       --realtime / --hold はデバッグ画面 (io.web_port) の確認用.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "omnivla/app.hpp"
#include "omnivla/controller.hpp"
#include "omnivla/tracker.hpp"

using nlohmann::json;
using namespace omnivla;

namespace {

json read_stdin() {
  std::string s((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  return json::parse(s);
}

std::optional<Pose> pose_of(const json& j) {
  if (j.is_null()) return std::nullopt;
  return Pose{j[0].get<double>(), j[1].get<double>(), j[2].get<double>()};
}

json optj(const std::optional<double>& v) { return v ? json(*v) : json(nullptr); }

int run_controller() {
  const json in = read_stdin();
  json out = json::array();
  for (const auto& c : in.at("cases")) {
    NavConfig nc = nav_config_from_json({{"engine", {{"controller", c.value("cfg", json::object())}}}});
    Waypoints w;
    for (const auto& p : c.at("waypoints")) w.push_back({p[0], p[1], p[2], p[3]});
    auto [v, om] = compute_command(w, nc.engine.controller);
    out.push_back({v, om});
  }
  std::cout << out.dump() << std::endl;
  return 0;
}

int run_tracker() {
  const json in = read_stdin();
  const TopomapIndex tm = topomap_index_from_json(in.at("topomap"));
  NavConfig nc = nav_config_from_json({{"engine", {{"tracker", in.value("cfg", json::object())}}}});
  const std::string mode = resolve_mode(nc.engine.tracker.reach_check, in.value("has_localization", false),
                                        in.value("has_odom", false), tm);
  SubgoalTracker tr(tm, nc.engine.tracker, mode);
  json steps = json::array();
  for (const auto& s : in.at("steps")) {
    std::function<double(int)> sim;
    std::map<int, double> sims;
    if (s.contains("sims") && !s["sims"].is_null()) {
      for (const auto& kv : s["sims"]) sims[kv[0].get<int>()] = kv[1].get<double>();
      sim = [&sims](int j) { return sims.count(j) ? sims[j] : 0.0; };
    }
    std::optional<double> travel;
    if (s.contains("travel") && !s["travel"].is_null()) travel = s["travel"].get<double>();
    const bool adv = tr.update(pose_of(s.value("pose", json(nullptr))), sim, travel);
    steps.push_back({{"advanced", adv}, {"index", tr.index}, {"done", tr.done}, {"reason", tr.last_reason},
                     {"similarity", optj(tr.last_similarity)}, {"distance", optj(tr.last_distance)}});
  }
  std::cout << json({{"mode", mode}, {"steps", steps}}).dump() << std::endl;
  return 0;
}

int run_replay(int argc, char** argv) {
  AppOptions opt;
  std::string frames_path;
  bool realtime = false;
  double hold = 0.0;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--repo") opt.repo_root = next();
    else if (a == "--frames") frames_path = next();
    else if (a == "--topomap") opt.topomap = next();
    else if (a == "--robot") opt.robot_config = next();
    else if (a == "--nav") opt.nav_config = next();
    else if (a == "--set") opt.sets.push_back(next());
    else if (a == "--realtime") realtime = true;
    else if (a == "--hold") hold = std::stod(next());
    else throw std::runtime_error("unknown option " + a);
  }
  std::ifstream f(frames_path);
  if (!f) throw std::runtime_error("cannot read " + frames_path);
  const json frames = json::parse(f);
  double now = 0.0;
  NavigatorApp app(opt, [&now] { return now; }, [](const std::string& s) { std::cerr << "[core_check] " << s << "\n"; },
                   /*threaded=*/false);
  auto& runner = app.runner();
  auto& eng = runner.engine();
  json out = json::array();
  bool started = false;
  uint64_t seq = 0;
  const auto wall0 = std::chrono::steady_clock::now();
  const double t0 = frames.at("frames").at(0).at("t").get<double>();
  for (const auto& fr : frames.at("frames")) {
    const double t = fr.at("t").get<double>();
    now = t;
    if (realtime) std::this_thread::sleep_until(wall0 + std::chrono::duration<double>(t - t0));
    auto frame = std::make_shared<Frame>();
    frame->key = "r" + std::to_string(seq++);
    frame->stamp = t;
    frame->kind = "encoded";
    frame->format = "jpeg";
    frame->data = read_file(fr.at("path").get<std::string>());
    const auto odom = pose_of(fr.value("odom", json(nullptr)));
    const auto loc = pose_of(fr.value("loc", json(nullptr)));
    if (!started) {  // desk_eval と同じ: 走り出す前にオドメトリ・自己位置を入れる
      if (odom) eng.on_odom(t, *odom);
      if (loc) eng.on_localization(t, *loc);
      if (!runner.start()) return 1;
      started = true;
    }
    eng.on_image(t, frame);
    if (odom) eng.on_odom(t, *odom);
    if (loc) eng.on_localization(t, *loc);
    json row = {{"t", t}};
    if (eng.state() == "running") {
      auto r = runner.step_once();
      if (r) {
        json wps = json::array();
        if (r->waypoints)
          for (const auto& p : *r->waypoints) wps.push_back({p[0], p[1], p[2], p[3]});
        row["state"] = r->state;
        row["subgoal"] = r->subgoal;
        row["v"] = r->v;
        row["w"] = r->w;
        row["waypoints"] = wps;
        row["similarity"] = optj(r->similarity);
        row["reason"] = r->reason;
        row["events"] = r->events;
      }
    } else {
      row["state"] = eng.state();
    }
    app.on_command(t, runner.command());
    out.push_back(row);
  }
  json result = {{"mode", eng.mode()}, {"frames", out}};
  std::cout << result.dump() << std::endl;
  if (hold > 0) std::this_thread::sleep_for(std::chrono::duration<double>(hold));
  runner.stop("replay finished");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: core_check controller|tracker|replay ...\n";
    return 2;
  }
  const std::string mode = argv[1];
  try {
    if (mode == "controller") return run_controller();
    if (mode == "tracker") return run_tracker();
    if (mode == "replay") return run_replay(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "core_check: " << e.what() << "\n";
    return 1;
  }
  std::cerr << "unknown mode " << mode << "\n";
  return 2;
}
