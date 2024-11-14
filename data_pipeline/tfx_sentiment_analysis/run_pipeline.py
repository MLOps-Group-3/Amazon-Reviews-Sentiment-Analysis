from absl import app
from pipeline.pipeline import run_pipeline

def main(argv):
    del argv  # Unused.
    run_pipeline()

if __name__ == '__main__':
    app.run(main)
