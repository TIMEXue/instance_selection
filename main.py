from datasets.base import DatasetBase
from algorithms.CHC_adaptive_search.chc import CHC
from algorithms.CHC_adaptive_search.chc_gallery import CHC_gallery
from configs import cfg

if __name__ == '__main__':
    pen_based_recognition = DatasetBase(train_path='datasets/awl_face_reid/gallery.txt',
                                        test_path='datasets/awl_face_reid/query.txt')

    chc = CHC_gallery(n_generation=cfg['EVOLUTION']['n_generation'],
                      n_population=cfg['EVOLUTION']['n_population'],
                      divergence_rate=cfg['EVOLUTION']['divergence_rate'],
                      class_number=cfg['EVOLUTION']['class_number'],
                      limit_sample=cfg['EVOLUTION']['limit_sample'],
                      X_data=pen_based_recognition.train_X,
                      y_data=pen_based_recognition.train_y,
                      alpha=cfg['EVOLUTION']['alpha'])

    chc.evolve()

    # print(pen_based_recognition.train_X.shape)
    # print(pen_based_recognition.train_y.shape)
    # print(pen_based_recognition.test_X.shape)
    # print(pen_based_recognition.test_y.shape)

