#pragma once
#include <time.h>
#include <chrono>
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
		_hh = rest / (3600 * 1000);
		rest = rest - _hh * 3600 * 1000;
		_mm = rest / (60 * 1000);
		rest = rest - _mm * 60 * 1000;
		_ss = rest / 1000;
		_ms = rest - _ss * 1000;
	}

	int InMilliseconds()
	{
		return _inMilliseconds;
	}

	double InSeconds()
	{
		return (double)_inMilliseconds / 1000.0;
	}

	friend ostream& operator<<(ostream& os, const Duration& d)
	{
		stringstream ss;
		if (d._hh < 10)
			ss << "0";
		ss << d._hh << ":";
		if (d._mm < 10)
			ss << "0";
		ss << d._mm << ":";
		if (d._ss < 10)
			ss << "0";
		ss << d._ss << ".";
		if (d._ms < 100)
			ss << "0";
		if (d._ms < 10)
			ss << "0";
		ss << d._ms;
		os << ss.str();
		return os;
	}
};

class Timer
{
private:
	bool _isPaused = false;
	Timer* _pauseTimer = nullptr;

	clock_t _cpu_start;
	clock_t _cpu_stop;

	chrono::time_point<chrono::high_resolution_clock> _elapsed_start;
	chrono::time_point<chrono::high_resolution_clock> _elapsed_stop;
public:
	Timer() {}

	void Start()
	{
		if (!_isPaused)
		{
			_cpu_start = clock();
			_elapsed_start = chrono::high_resolution_clock::now();
		}
		else
		{
			_pauseTimer->Stop();
			_cpu_start = _pauseTimer->_cpu_stop - (_cpu_stop - _cpu_start);
			_isPaused = false;
		}
	}

	void Pause()
	{
		if (!_pauseTimer)
			_pauseTimer = new Timer();
		this->Stop();
		_pauseTimer->Start();
		_isPaused = true;
	}

	void Stop()
	{
		_cpu_stop = clock();
		_elapsed_stop = chrono::high_resolution_clock::now();
	}

	Duration CPU() const
	{
		double span = (double)(_cpu_stop - _cpu_start);
		Duration d((int)((double)span / CLOCKS_PER_SEC * 1000));
		return d;
	}

	Duration Elapsed() const
	{
		double durationInMilliseconds = chrono::duration_cast<chrono::duration<double, std::milli>>(_elapsed_stop - _elapsed_start).count();
		Duration d(durationInMilliseconds);
		return d;
	}

	~Timer()
	{
		if (_pauseTimer)
			delete _pauseTimer;
	}
};