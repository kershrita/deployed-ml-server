from src.model_trainer import ModelTrainer

if __name__ == '__main__':
    trainer = ModelTrainer(model_path='models/classification_model.pkl')
    trainer.train(data_path='data/data.csv')
