import numpy as np

class ConcordanceCorrelationCoefficient:

    def __call__(self, observed, predicted):
        mean_observed = np.mean(observed)
        mean_predicted = np.mean(predicted)
        covariance = np.cov(observed, predicted, bias=True)[0][1]
        # ddof 是 “delta degrees of freedom”。
        # 默认 ddof=0 计算的是总体方差（除以 N），而 ddof=1 相当于无偏样本方差（除以 N-1），用来在样本较小时估计总体方差更准确。
        # 这里设为 1，是把 observed、predicted 看作样本而不是完整总体。
        obs_variance = np.var(observed, ddof=1)
        pred_variance = np.var(predicted, ddof=1)
        denominator = obs_variance + pred_variance + (mean_observed - mean_predicted) ** 2
        return 2 * covariance / denominator if denominator != 0 else 0.0