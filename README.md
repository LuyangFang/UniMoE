# Generalizable and Efficient Automated Scoring with a Knowledge-Distilled Multi-Task Mixture-of-Experts

**This repository contains code for the paper "Generalizable and Efficient Automated Scoring with a Knowledge-Distilled Multi-Task Mixture-of-Experts".**


## Usage Example
### Example Teacher Fine-tuning Usage:
python finetune_gptoss20b_lora_jsonl_cls.py \
    --model_name "openai/gpt-oss-20b" \
    --train_file ./data/task1_train.csv \
    --test_file ./data/task1_test.csv \
    --max_length 512 \
    --batch_size 4 \
    --accum_steps 8 \
    --epochs 10 \
    --lr 2.0e-5 \
    --warmup_ratio 0.10 \
    --weight_decay 0.02 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --gradient_checkpointing \
    --bf16 \
    --log_every_steps 200 \
    --max_grad_norm 0.3 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --threshold 0.5 \
    --early_stop_patience 4 \
    --embed_dim 768 \
    --output_dir ./results/task1 \
    --probs_out ./results/task1/train_probs.csv \
    --include_text_in_csv \
    > ./results/task1/train.log \
    2> ./results/task1/train.err

### MTL_MOE Example:
python mtl_moe_trainer.py --num_experts 4 --kd_weight 0.5 --load_balance_weight 0.01 --seed 42 --max_epochs 20 --patience 3 --data_dir '../PASTA_data/new_processed_data/' --output_dir './PASTA-Models/'




## Dataset Overview

This research utilizes pre-existing datasets, incorporating responses from middle school students that have been evaluated by experts for nine multi-label assessment tasks from the PASTA project [1][2]. These assessment tasks are specifically crafted to evaluate middle school students' ability to apply multi-label knowledge in explaining scientific phenomena. The NGSS framework aims to facilitate students in developing applied knowledge across educational levels by integrating disciplinary core ideas (DCIs), crosscutting concepts (CCCs), and science and engineering practices (SEPs) within K-12 performance expectations.

The assessment tasks in this study align with the NGSS middle school-level expectations: students must analyze and interpret data to determine whether substances possess identical properties [3]. This expectation requires students to employ knowledge of the structure and properties of matter, chemical reactions (DCIs), and patterns (CCC) to effectively analyze and interpret data (SEP).

![Illustrative Multi-label Task: Gas-Filled Balloons](gas_filled_ballon.png)  
*Figure: Illustrative Multi-label Task – Gas-Filled Balloons*

A total of 1,200 students in grades 6–8 participated in this study. Middle school teachers across the U.S. invited their students to engage with open-ended NGSS-aligned science tasks [4]. After data cleaning, fewer than 1,200 responses remained per task (exact counts in the table below). Responses were randomly selected to form training, validation, and test sets for machine learning models. For privacy, all data was anonymized, and demographic details were unavailable. Nonetheless, due to the geographical diversity of participating teachers, the dataset is considered representative of the broader US middle school student population.

The assessment tasks were sourced from the Next Generation Science Assessment [1] and required students to apply fundamental *chemistry* principles to real-world contexts. Falling under the physical sciences domain, specifically "Matter and its Characteristics," these tasks assess students' ability to analyze data and differentiate substances by their attributes. These tasks were designed to assess students' multi-dimensional thinking and provide educators with insights that could inform instructional strategies.

Automated reports derived from rubric-based scoring highlight topics where students may require additional support. For instance, in one task, students were asked to identify gases in an experiment by comparing their properties to those documented in a data table (refer to the figure above). Successfully completing this task required understanding the structure and properties of matter, chemical reactions, and the ability to plan investigations while recognizing patterns in the data.

A structured scoring rubric was developed to encompass five response dimensions, corresponding to the science learning framework: SEP+DCI, SEP+CCC, SEP+CCC, DCI, and DCI. This rubric was designed to capture multi-dimensional cognitive processes [5]. The table below outlines the specific criteria for each category. Students were assessed simultaneously across multiple perspectives, receiving scores that reflected their understanding of DCIs, CCCs, and SEPs as aligned with the rubric. To enhance the validity of these multi-perspective rubrics, the research team collaborated with experienced science educators.

### Scoring Rubric for Task: Gas-Filled Balloons (Task 5)

| ID  | Perspective | Description |
|-----|-------------|-------------|
| E1  | SEP+DCI     | Student states that Gas A and D could be the same substance. |
| E2  | SEP+CCC     | Student describes the pattern (comparing data in different columns) in the table flammability data of Gas A and Gas D as the same. |
| E3  | SEP+CCC     | Student describes the pattern (comparing data in different columns) in density data of Gas A and Gas D, which is the same in the table. |
| E4  | DCI         | Student indicates flammability is one characteristic of identifying substances. |
| E5  | DCI         | Student indicates density is one characteristic of identifying substances. |

### Dataset Information for Multi-label and Multi-class Tasks

| ID     | Item                          | No. Labels | Training Size | Testing Size |
|--------|-------------------------------|------------|----------------|---------------|
| Task 1 | Anna vs Carla                 | 4          | 955            | 239           |
| Task 2 | Breaking Down Hydrogen Peroxide | 4        | 666            | 167           |
| Task 3 | Carlos Javier Atomic Model    | 5          | 956            | 240           |
| Task 4 | Dry Ice Model                 | 3          | 1111           | 278           |
| Task 5 | Gas Filled Balloon            | 3          | 958            | 240           |
| Task 6 | Layers in Test Tube           | 10         | 956            | 240           |
| Task 7 | Model For Making Water        | 5          | 836            | 210           |
| Task 8 | Nami Careful Experiment       | 6          | 653            | 164           |
| Task 9 | Natural Sugar                 | 5          | 956            | 239           |

---






