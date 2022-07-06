

#ifndef GPUTIMER_H_
#define GPUTIMER_H_


class GPUTimer
{
    public:
        GPUTimer();
        virtual ~GPUTimer();

        void start_timer(cudaStream_t stream_id = 0);
        void stop_timer(cudaStream_t stream_id = 0);
        void print_elapsed_time();

    public:
        bool m_Started;
        bool m_Stopped;
        cudaEvent_t m_StartEvent;
        cudaEvent_t m_StopEvent;
};
