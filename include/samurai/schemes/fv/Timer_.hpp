#pragma once
#include <chrono>
#include <time.h>
using namespace std;

class Duration
{
  private:

    int _inMilliseconds;
    int _hh;
    int _mm;
    int _ss;
    int _ms;

  public:

    Duration(int durationInMilliseconds)
    {
        _inMilliseconds = durationInMilliseconds;

        int rest = durationInMilliseconds;
        _hh      = rest / (3600 * 1000);
        rest     = rest - _hh * 3600 * 1000;
        _mm      = rest / (60 * 1000);
        rest     = rest - _mm * 60 * 1000;
        _ss      = rest / 1000;
        _ms      = rest - _ss * 1000;
    }

    int InMilliseconds()
    {
        return _inMilliseconds;
    }

    double InSeconds()
    {
        return static_cast<double>(_inMilliseconds) / 1000.0;
    }

    friend ostream& operator<<(ostream& os, const Duration& d)
    {
        stringstream ss;
        if (d._hh < 10)
        {
            ss << "0";
        }
        ss << d._hh << ":";
        if (d._mm < 10)
        {
            ss << "0";
        }
        ss << d._mm << ":";
        if (d._ss < 10)
        {
            ss << "0";
        }
        ss << d._ss << ".";
        if (d._ms < 100)
        {
            ss << "0";
        }
        if (d._ms < 10)
        {
            ss << "0";
        }
        ss << d._ms;
        os << ss.str();
        return os;
    }
};

enum TimerState
{
    None,
    Started,
    Paused,
    Stopped
};

class Timer
{
  private:

    TimerState _state = TimerState::Stopped;

    clock_t _cpu_start;
    clock_t _cpu_stop;
    clock_t _cpu_pause_start;

    chrono::time_point<chrono::high_resolution_clock> _elapsed_start;
    chrono::time_point<chrono::high_resolution_clock> _elapsed_stop;

  public:

    Timer()
    {
        _cpu_start = clock();
        _cpu_stop  = clock();
    }

    void Start()
    {
        if (_state == TimerState::Stopped)
        {
            _cpu_start     = clock();
            _elapsed_start = chrono::high_resolution_clock::now();
        }
        else if (_state == TimerState::Paused)
        {
            _cpu_start += clock() - _cpu_pause_start;
        }
        _state = TimerState::Started;
    }

    inline void Pause()
    {
        if (_state == TimerState::Started)
        {
            _cpu_pause_start = clock();
            _state           = TimerState::Paused;
        }
    }

    inline void Stop()
    {
        if (_state != TimerState::Stopped)
        {
            _cpu_stop     = _state == TimerState::Paused ? _cpu_pause_start : clock();
            _elapsed_stop = chrono::high_resolution_clock::now();
            _state        = TimerState::Stopped;
        }
    }

    Duration CPU()
    {
        if (_state != TimerState::Stopped)
        {
            _cpu_stop     = _state == TimerState::Paused ? _cpu_pause_start : clock();
            _elapsed_stop = chrono::high_resolution_clock::now();
        }
        auto span   = _cpu_stop - _cpu_start;
        int span_ms = int(static_cast<double>(span) / CLOCKS_PER_SEC * 1000);
        // std::cout << "start " << _cpu_start << ", stop " << _cpu_stop << ", span " << span << ", ms " << CLOCKS_PER_SEC << ", span_ms "
        // << span_ms << "  ";
        Duration d(span_ms);
        return d;
    }

    Duration Elapsed() const
    {
        double durationInMilliseconds = chrono::duration_cast<chrono::duration<double, std::milli>>(_elapsed_stop - _elapsed_start).count();
        Duration d(int(round(durationInMilliseconds)));
        return d;
    }
};
