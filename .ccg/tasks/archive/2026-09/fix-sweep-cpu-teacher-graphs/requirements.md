# CPU-safe Stage 1 teacher graphs

Pass the device selected by `train_stage1` through `build_teacher_graphs` into both `mutual_knn` calls. Add a small synthetic test that explicitly requests CPU. Preserve seed ordering for a later task.
