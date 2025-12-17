from src.trainer import ModelTrainer

def main():
    print("=== DEVELOPER MODE: TRAINING AI ===")
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()