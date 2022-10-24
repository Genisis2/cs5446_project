This README serves as the ingredients of the final report, Monte Carlo learning section. 

Two steps:
1. compute utility of states
    - split the match to multiple rallies based on `is_final_shot`
    - for each rally, split to two trails based on `hit_id`, so one trail will focus on state transition of single `hit_id`
    - compute expected utility of each state (page 17,18 of Lecture 7)
        - stroke with `is_final_shot=false` is -0.4 (tentatively set as -0.4)
        - final shot is 1 if `final_outcome=1` else -1
2. supervised learning - standard regression problem
    - ? 

Step 2 is the part that under discussion, Jihui and I had some brief discussion but we still cannot come up with a way to frame this step. 