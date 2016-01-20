#ifndef TIMING_STATISTICS_H_
#define TIMING_STATISTICS_H_

#include <list>

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
    float getStdDev() { return sqrtf(variance)/float(count); }
    float calculateSpeedup(float reference) {
        return reference / average;
    }
    float calculateSpeedup(TimingStatistics & reference) {
        return reference.getAverage() / average;
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

#endif
