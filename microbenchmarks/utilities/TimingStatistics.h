#ifndef TIMING_STATISTICS_H_
#define TIMING_STATISTICS_H_

#include <list>
#include <cmath>

class TimingStatistics {
private:
    std::list<unsigned long long> measurements;
    float average, variance;
    int count;

public:

    TimingStatistics() {
        average = 0.0f;
        variance = 0.0f;
        count = 0;
    }

    ~TimingStatistics() {
        measurements.clear();
    }

    void update(unsigned long long elapsedTime) {
        measurements.push_back(elapsedTime);
        float delta = float(elapsedTime) - average;
        average += delta / (1.0f + float(count));
        variance += delta * (float(elapsedTime) - average);

        count++;
    }

    float getAverage() { return average; }
    float getStdDev() { return count > 0 ? sqrtf(variance)/float(count) : 0.0f; }
    float calculateSpeedup(float reference) {
        return reference / average;
    }
    float calculateSpeedup(TimingStatistics & reference) {
        return average > 0.0f ? reference.getAverage() / average : 0.0f;
    }

    void printList() {
        std::list<unsigned long long>::iterator iter;
        for (iter = measurements.begin(); iter != measurements.end(); iter++) {
            std::cout << (*iter) << std::endl;
        }
    }

    // Calculate 90% confidence level. Adding/subtracting this
    // value from average will give upper/lower bounds.
    float confidence90() {
        return 1.645f * getStdDev() / sqrtf(float(count));
    }

    // Calculate 95% confidence level. Adding/subtracting this
    // value from average will give upper/lower bounds.
    float confidence95() {
        return 1.96f * getStdDev() / sqrtf(float(count));
    }
};


template<typename IN_TYPE>
class Statistics {
private:
    typename std::list<IN_TYPE> measurements;
    double average, variance;
    int count;

public:

    Statistics() {
        average = 0.0;
        variance = 0.0;
        count = 0;
    }

    ~Statistics() {
        measurements.clear();
    }

    void update(IN_TYPE elapsedTime) {
        measurements.push_back(elapsedTime);
        double delta = double(elapsedTime) - average;
        average += delta / (1.0 + double(count));
        variance += delta * (double(elapsedTime) - average);

        count++;
    }

    double getAverage() { return average; }
    double getStdDev() { return std::sqrt(variance) / double(count); }
    double calculateSpeedup(double reference) {
        return reference / average;
    }
    double calculateSpeedup(TimingStatistics & reference) {
        return reference.getAverage() / average;
    }

    void printList() {
        typename std::list<IN_TYPE>::iterator iter;
        for (iter = measurements.begin(); iter != measurements.end(); iter++) {
            std::cout << (*iter) << std::endl;
        }
    }

    // Calculate 90% confidence level. Adding/subtracting this
    // value from average will give upper/lower bounds.
    double confidence90() {
        return 1.645 * getStdDev() / sqrt(double(count));
    }

    // Calculate 95% confidence level. Adding/subtracting this
    // value from average will give upper/lower bounds.
    float confidence95() {
        return 1.96 * getStdDev() / sqrt(double(count));
    }
};

#include <chrono>
static inline unsigned long long get_timestamp(void)
{
    auto t0 = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(t0);
    return t1.count();
}


typedef unsigned long long TIMING_RES;

#endif
