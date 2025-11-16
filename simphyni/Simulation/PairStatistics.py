import numpy as np
from sklearn.metrics import mutual_info_score

class PairStatistics:

    @staticmethod
    def count_statistic(trait1: np.ndarray, trait2: np.ndarray) -> np.ndarray:
        return np.sum(np.logical_and(trait1, trait2), axis=0)

    @staticmethod
    def _vectorized_pair_statistic(trait1: np.ndarray, trait2: np.ndarray) -> np.ndarray:
        cooc = np.sum(np.logical_and(trait1, trait2), axis=0)
        sum_trait1 = np.sum(trait1 > 0, axis=0)
        sum_trait2 = np.sum(trait2 > 0, axis=0)
        oe_ratio = cooc / (sum_trait1 * sum_trait2) * trait1.shape[0]
        epsilon = - (sum_trait1 * sum_trait2) / (trait1.shape[0] ** 2)
        oe_ratio = np.nan_to_num(oe_ratio, nan=0)
        return oe_ratio + epsilon
    
    @staticmethod
    def _log_alpha_vectorized_pair_statistic(trait1: np.ndarray, trait2: np.ndarray) -> np.ndarray:
        cooc = np.sum(np.logical_and(trait1, trait2), axis=0)
        sum_trait1 = np.sum(trait1 > 0, axis=0)
        sum_trait2 = np.sum(trait2 > 0, axis=0)
        oe_ratio = cooc / (sum_trait1 * sum_trait2) * trait1.shape[0]
        alpha = 0.0001
        epsilon = 1 - (sum_trait1 * sum_trait2) / (trait1.shape[0] ** 2) * alpha
        oe_ratio = np.nan_to_num(oe_ratio, nan=0)
        return np.log(oe_ratio + epsilon)
    
    @staticmethod
    def _chi_pair_statistic(trait1: np.ndarray, trait2: np.ndarray) -> np.ndarray:
        obs = np.sum(np.logical_and(trait1, trait2), axis=0)
        sum_trait1 = np.sum(trait1 > 0, axis=0)
        sum_trait2 = np.sum(trait2 > 0, axis=0)
        exp = (sum_trait1 * sum_trait2) / trait1.shape[0]
        chi = (obs - exp) ** 2 / exp
        return chi

    @staticmethod
    def _oe_pair_statistic(trait1: np.ndarray, trait2: np.ndarray) -> np.ndarray:
        cooc = np.sum(np.logical_and(trait1, trait2), axis=0)
        sum_trait1 = np.sum(trait1 > 0, axis=0)
        sum_trait2 = np.sum(trait2 > 0, axis=0)
        oe_ratio = cooc / (sum_trait1 * sum_trait2) * trait1.shape[0]
        oe_ratio = np.nan_to_num(oe_ratio, nan=0)
        return oe_ratio

    @staticmethod
    def _jaccard_index_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        intersection = np.logical_and(tp, tq).sum(axis=0)
        union = np.logical_or(tp, tq).sum(axis=0)
        return intersection / np.maximum(union, 1)
    
    @staticmethod
    def _log_jaccard_index_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        intersection = np.logical_and(tp, tq).sum(axis=0)
        union = np.logical_or(tp, tq).sum(axis=0)
        return np.log((intersection + 0.00001 / np.maximum(union, 1)))

    @staticmethod
    def _modified_jaccard_index_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        intersection = np.logical_and(tp, tq).sum(axis=0)
        union = np.logical_or(tp, tq).sum(axis=0)
        epsilon = - tp.sum(axis = 0) * tq.sum(axis = 0) / (tp.shape[0]**2)
        return intersection / np.maximum(union, 1) + epsilon

    @staticmethod
    def _pearson_correlation_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        tp_mean = np.mean(tp, axis=0)
        tq_mean = np.mean(tq, axis=0)
        tp_std = np.std(tp, axis=0, ddof=0)
        tq_std = np.std(tq, axis=0, ddof=0)

        # Subtract mean to center the data
        tp_centered = tp - tp_mean
        tq_centered = tq - tq_mean

        # Calculate covariance for each trial
        covariance = np.sum(tp_centered * tq_centered, axis=0)

        # Calculate Pearson correlation for each trial
        pearson_values = covariance / (tp.shape[0] * tp_std * tq_std)

        # Handle cases where variance is zero
        if len(tq.shape) > 1:
            pearson_values[tp_std == 0] = 0
            pearson_values[tq_std == 0] = 0

        return pearson_values

    @staticmethod
    def _odds_ratio_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        epsilon = 0.01
        a = np.logical_and(tp, tq).sum(axis=0) + epsilon # Both traits present
        b = np.logical_and(tp, np.logical_not(tq)).sum(axis=0)  + epsilon# First trait present, second absent
        c = np.logical_and(np.logical_not(tp), tq).sum(axis=0) + epsilon # First trait absent, second present
        d = np.logical_and(np.logical_not(tp), np.logical_not(tq)).sum(axis=0) + epsilon # Both traits absent

        # Calculate odds ratio; avoid division by zero
        # odds_ratio_values = (a * d) / np.maximum(b * c, 1)
        odds_ratio_values = (a * d) / (b * c)

        return odds_ratio_values
    
    @staticmethod
    def _treewas_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        a = np.logical_and(tp, tq).sum(axis=0)# Both traits present
        b = np.logical_and(tp, np.logical_not(tq)).sum(axis=0)# First trait present, second absent
        c = np.logical_and(np.logical_not(tp), tq).sum(axis=0) # First trait absent, second present
        d = np.logical_and(np.logical_not(tp), np.logical_not(tq)).sum(axis=0) # Both traits absent
        score = a + d - c - b
        return score

    @staticmethod
    def _log_odds_ratio_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        epsilon = 1#e-2
        a = np.logical_and(tp, tq).sum(axis=0) + epsilon # Both traits present
        b = np.logical_and(tp, np.logical_not(tq)).sum(axis=0)  + epsilon# First trait present, second absent
        c = np.logical_and(np.logical_not(tp), tq).sum(axis=0) + epsilon # First trait absent, second present
        d = np.logical_and(np.logical_not(tp), np.logical_not(tq)).sum(axis=0) + epsilon # Both traits absent

        # Calculate odds ratio; avoid division by zero
        # odds_ratio_values = (a * d) / np.maximum(b * c, 1)
        odds_ratio_values = (a * d) / (b * c)

        return np.log(odds_ratio_values)
    
    @staticmethod
    def _log_add_ratio_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        epsilon = 1e-2
        a = np.logical_and(tp, tq).sum(axis=0) + epsilon # Both traits present
        b = np.logical_and(tp, np.logical_not(tq)).sum(axis=0)  + epsilon# First trait present, second absent
        c = np.logical_and(np.logical_not(tp), tq).sum(axis=0) + epsilon # First trait absent, second present
        d = np.logical_and(np.logical_not(tp), np.logical_not(tq)).sum(axis=0) + epsilon # Both traits absent

        # Calculate odds ratio; avoid division by zero
        # odds_ratio_values = (a * d) / np.maximum(b * c, 1)
        odds_ratio_values = (a + d) / (b + c)

        return np.log(odds_ratio_values)

    @staticmethod
    def _mutual_information_statistic(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        if len(tp.shape) == 1:
            mutual_information_values = np.array([
                mutual_info_score(tp, tq)
            ])
            return mutual_information_values
        mutual_information_values = np.array([
            mutual_info_score(tp[:, trial], tq[:, trial]) for trial in range(tp.shape[1])
        ])
        return mutual_information_values

    
    @staticmethod
    def cosine_similarity(trait1: np.ndarray, trait2: np.ndarray) -> np.ndarray:
        dot_product = np.sum(trait1 * trait2, axis=0)
        norm_trait1 = np.linalg.norm(trait1, axis=0)
        norm_trait2 = np.linalg.norm(trait2, axis=0)
        return dot_product / (norm_trait1 * norm_trait2)
    
    @staticmethod
    def z_statistic(trait1, trait2):
        cooc = np.sum(np.logical_and(trait1, trait2), axis=0)
        sum_trait1 = np.sum(trait1 > 0, axis=0)
        sum_trait2 = np.sum(trait2 > 0, axis=0)
        exp = sum_trait1 * sum_trait2/ trait1.shape[0]
        epsilon = 0.0001
        p = sum_trait1 * sum_trait2/ trait1.shape[0]**2 
        if type(p) != np.ndarray:
            p = min(max(p,epsilon),1-epsilon)
        else:
            p[p == 1] = 1 - epsilon
            p[p == 0] = 0 + epsilon
        # return np.nan_to_num((cooc - exp)/(p * (1-p)),nan = 0)
        return (cooc - exp)/(p * (1-p))
    