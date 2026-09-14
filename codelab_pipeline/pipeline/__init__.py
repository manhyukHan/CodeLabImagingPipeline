"""The pipeline runner: what a store already holds per FOV and step,
what a policy asks to run, and (through the app's own batch actions)
the run itself.

    status.py   Qt-free: the project from its config, the per-FOV state
                of every step read from the store, and the plan a
                policy implies
    __main__    the CLI: `python -m codelab_pipeline.pipeline status
                <config.xml>` / `plan ...`

The execution half (the sequencer over MainWindow's batch actions and
the tab) lives in windows/pipeline_wiring.py and ui/pipeline_panel.py,
because five of the steps only exist as app actions.
"""
