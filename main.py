from datasets.base import DatasetBase
from algorithms.CHC_adaptive_search.chc import CHC
from configs import cfg

if __name__ == '__main__':
    pen_based_recognition = DatasetBase(train_path='datasets/raw_data/pendigits.tra',
                                        test_path='datasets/raw_data/pendigits.tes')

    chc = CHC(n_generation=cfg['EVOLUTION']['n_generation'],
              n_population=cfg['EVOLUTION']['n_population'],
              divergence_rate=cfg['EVOLUTION']['divergence_rate'],
              X_data=pen_based_recognition.train_X,
              y_data=pen_based_recognition.train_y,
              alpha=cfg['EVOLUTION']['alpha'])

    chc.evolve()

    # print(pen_based_recognition.train_X.shape)
    # print(pen_based_recognition.train_y.shape)
    # print(pen_based_recognition.test_X.shape)
    # print(pen_based_recognition.test_y.shape)

