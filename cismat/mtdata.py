import numpy as np
from numpy.linalg import norm
import itertools


class MTData:
    def __init__(self, df):
        self.df = df

    def generate_similarity_scores(
        self, fun=lambda x, y: np.dot(x, y) / (norm(x) * norm(y)), pairs=None
    ):
        sim_dict = {id: {} for id in self.df}
        if pairs is None:
            for id1 in self.df:
                images1 = self.df[id1]
                for id2 in self.df:
                    images2 = self.df[id2]
                    sim_list = []
                    if id1 == id2:
                        combs = itertools.combinations(images1, 2)
                    else:
                        combs = itertools.product(images1, images2)
                    for img1, img2 in combs:
                        if (id1 == id2 and img1 != img2) or id1 != id2:
                            sim_list.append(fun(images1[img1], images2[img2]))
                    sim_dict[id1][id2] = sim_list
                    sim_dict[id2][id1] = sim_list
        else:
            # Compute similarities for specified image pairs
            for id1, img1, id2, img2 in pairs:
                score = fun(self.df[id1][img1], self.df[id2][img2])
                sim_dict[id1].setdefault(id2, []).append(score)
                sim_dict[id2].setdefault(id1, []).append(score)

            sim_dict = {id: sim_dict[id] for id in sim_dict if sim_dict[id]}

        self.similarity_scores = sim_dict
