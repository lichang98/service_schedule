/// This file is the util for file processing

#include <cstdio>
#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace utils
{
    const static char WORK_ORDER[100] = "../data/SIC_round1_testA_20200909/work_order.csv";
    const static char PROCESS_TM_MAT[100] = "../data/SIC_round1_testA_20200909/process_time_matrix.csv";
    const static char PRED_RESULT[100] = "../prediction_result/result.csv";
    size_t PROCESS_TM_FILE_LINE_MAXLEN = 1000;
    const static int EXPERT_MAX_PARALLEL = 3;
    const static int EXPERT_NOT_GOOD_TIME = 999999; // If expert not good at some type tasks, the processing time will be 999999

    struct Task
    {
        int task_id;
        int tm_stamp;
        int type;
        int max_resp_tm;

        Task() : task_id(-1), tm_stamp(-1), type(-1), max_resp_tm(-1) {}

        Task(int _task_id, int _tm_stamp, int _type, int _max_resp_tm) : task_id(_task_id), tm_stamp(_tm_stamp), type(_type), max_resp_tm(_max_resp_tm) {}

        friend std::ostream &operator<<(std::ostream &out, const Task &task)
        {
            out << "Task: ["
                << "task id=" << task.task_id << ", tm_stamp=" << task.tm_stamp << ", type=" << task.type << ", max_resp_tm=" << task.max_resp_tm << "]";
            return out;
        }
    };

    struct Expert
    {
        std::vector<int> process_dura;
        Task *process_tasks[EXPERT_MAX_PARALLEL];
        int process_remains[EXPERT_MAX_PARALLEL];
        int num_avail;

        Expert()
        {
            std::fill(process_tasks, process_tasks + EXPERT_MAX_PARALLEL, nullptr);
            std::fill(process_remains, process_remains + EXPERT_MAX_PARALLEL, 0);
            num_avail = EXPERT_MAX_PARALLEL;
        }

        friend std::ostream &operator<<(std::ostream &out, const Expert &e)
        {
            out << "Expert, time for " << e.process_dura.size() << " kinds of tasks, times are :[";
            for (int val : e.process_dura)
                out << val << ",";
            out << "]";
            return out;
        }

        int num_types()
        {
            return process_dura.size();
        }

        // The time elapsed one time slot, the expert process each task one time slot
        // Update the remains time
        std::vector<Task *> update()
        {
            std::vector<Task *> finish_tasks;
            for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
            {
                if (process_remains[i] ==0 && process_tasks[i])
                {
                    finish_tasks.push_back(process_tasks[i]);
                    process_tasks[i] = nullptr;
                    num_avail++;
                }
                else
                {
                    process_remains[i]--;
                }
            }
            return finish_tasks;
        }

        bool assign_task(Task *task)
        {
            if (num_avail <= 0)
                return false;
            int task_type = task->type;
            int process_time = process_dura[task_type];
            process_remains[EXPERT_MAX_PARALLEL - num_avail] = process_time;
            process_tasks[EXPERT_MAX_PARALLEL - num_avail] = task;
            num_avail--;
            return true;
        }
    };

    // Load work order content
    // Each line contains four integers
    // Task ID, Task generating time, type, max response time
    std::vector<Task> load_work_order()
    {
        std::vector<Task> tasks;
        FILE *fp = fopen(WORK_ORDER, "r");
        int task_id, tm, type, max_resp;
        while (fscanf(fp, " %d,%d,%d,%d", &task_id, &tm, &type, &max_resp) != EOF)
        {
            tasks.emplace_back(Task(task_id, tm, type - 1, max_resp)); // change type start from index 1 to 0
        }
        fclose(fp);
        return tasks;
    }

    std::vector<Expert> load_expert_process_duras()
    {
        std::vector<Expert> experts;
        FILE *fp = fopen(PROCESS_TM_MAT, "r");
        char *line = nullptr;
        getline(&line, &PROCESS_TM_FILE_LINE_MAXLEN, fp);
        while (getline(&line, &PROCESS_TM_FILE_LINE_MAXLEN, fp) != EOF)
        {
            Expert expert;
            char *e = strtok(line, ",");
            e = strtok(nullptr, ",");
            while (e != nullptr)
            {
                expert.process_dura.push_back(atoi(e));
                e = strtok(nullptr, ",");
            }
            experts.emplace_back(expert);
        }
        fclose(fp);
        return experts;
    }

} // namespace utils

// int main(int argc, char const *argv[])
// {
//     std::vector<Task> tasks = load_work_order();
//     std::vector<Expert> experts = load_expert_process_duras();
//     for (Expert e : experts)
//         std::cout << e << std::endl;
//     return 0;
// }
