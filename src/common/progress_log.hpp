#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace neuroevolution::common {

using ProgressClock = std::chrono::steady_clock;

inline std::string FormatCurrentLocalTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now_time);
#else
    localtime_r(&now_time, &local_time);
#endif

    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}

inline double ElapsedMilliseconds(const ProgressClock::time_point start_time,
                                  const ProgressClock::time_point end_time = ProgressClock::now()) {
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

inline void PrintTimestampedProgressLine(std::ostream &stream, const std::string_view message) {
    stream << '[' << FormatCurrentLocalTimestamp() << "] " << message << '\n' << std::flush;
}

inline void PrintTimestampedProgressDuration(std::ostream &stream, const std::string_view message,
                                             const ProgressClock::time_point start_time,
                                             const ProgressClock::time_point end_time = ProgressClock::now()) {
    std::ostringstream formatted_message;
    formatted_message << message << " in " << std::fixed << std::setprecision(1)
                      << ElapsedMilliseconds(start_time, end_time) << " ms";
    PrintTimestampedProgressLine(stream, formatted_message.str());
}

} // namespace neuroevolution::common
