# Isolation design

This is a design for a later stage. No trainer and no evaluator are implemented here.

## Processes

`graybox_train/` may import the Full model, the processor, page loading, the uint8 projection, and the Full native-cache Native-FDP term.

`target_fastv_evaluate/` is a separate directory and a separate Python process. It is the only place allowed to import the denylist. It reads sealed PNGs. It does not import the trainer, and the trainer does not import it.

The two sides do not share a process, a `sys.modules` entry, a GPU cache, or a Python object.

## Static check

Before a future trainer starts, an import scanner walks `graybox_train/` and fails if any file imports a denylist module or contains `install(` with `"fastv"`. The scanner result is written next to the run config. A failed scan means the run does not start.

## Runtime assertion

The trainer installs an import hook. If a denylist module name enters `sys.modules`, the hook increments `target_fastv_access_count` and appends the module name. The legal start state is:

`target_fastv_access_count == 0`

`target_fastv_modules_loaded == []`

The same pair is written at the end of training. Any nonzero count invalidates the run. The log is append-only and is hashed with the PNGs.

Calling `model.generate` is allowed only with FastV hooks absent. The trainer asserts `clear_hooks` was the last hook operation and that no forward pre-hook on language layer 3 is registered. That assertion is a check, not a FastV call.

## Seal

When the step budget ends, the trainer writes, in order:

1. the uint8 PNG for each sample;
2. the sha256 of each PNG;
3. the sha256 of the frozen config and of the access log;
4. the training log, including the two counters.

It then exits. The evaluator starts only after that process is gone and the hash file exists. The evaluator refuses a PNG whose bytes do not match the sealed hash. It does not resume the optimizer and does not return gradients to the trainer.

This design is auditable because the denylist, the import hook, the counters, and the process boundary are externally checkable. It is not implemented in this stage, so it has not been executed.
