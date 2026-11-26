"""Stop full-pool training at the registered budget and save the earlier crossing."""
import json
from experiment_callback import ExperimentCallback
from swift.callbacks import callbacks_map


class RoundThreeFull(ExperimentCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args,state,control,**kwargs)
        if self.token_goal != 12020912 or len(self.run['ids']) != 4000:
            raise ValueError('Full-pool tokenization differs from the registered pool')
        self.token_goal=3005228
        self.early_saved=False
        p=self.out/'runtime.json'
        info=json.loads(p.read_text())
        info.update(token_goal=self.token_goal,early_token_goal=1202092)
        p.write_text(json.dumps(info,indent=2)+'\n')

    def on_step_end(self,args,state,control,**kwargs):
        if not self.early_saved and self.input_count >= 1202092:
            self.save_steps.add(state.global_step)
            self.early_saved=True
        return super().on_step_end(args,state,control,**kwargs)


callbacks_map['round3_full']=RoundThreeFull
