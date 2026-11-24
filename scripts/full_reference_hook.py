"""Additional first-crossing checkpoint for the optional full-data reference only."""

import json

from swift.callbacks import callbacks_map
from experiment_callback import ExperimentCallback


class FullDataReference(ExperimentCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.matched = False
        self.primary_goal = self.run["matched_token_goal"]
        if not 0 < self.primary_goal < self.token_goal:
            raise ValueError("Expected primary budget strictly below full-data budget")

    def on_step_end(self, args, state, control, **kwargs):
        if not self.matched and self.input_count >= self.primary_goal:
            self.save_steps.add(state.global_step)
            self.matched = True
            control.should_save = True
            (self.out / "matched-step.json").write_text(
                json.dumps(
                    {
                        "step": state.global_step,
                        "target_tokens": self.primary_goal,
                        "actual_tokens": self.input_count,
                        "answer_tokens": self.answer_count,
                    }
                )
                + "\n"
            )
        return super().on_step_end(args, state, control, **kwargs)


callbacks_map["full_data_reference"] = FullDataReference
