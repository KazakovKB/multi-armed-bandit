import pandas as pd
import numpy as np
from scipy.stats import norm, invgamma

from tqdm import tqdm
from ipywidgets import widgets
from IPython.display import display

class normal_inverse_gamma:

    def __init__(self, arms_info: dict, rounds: int = 20000, baseline_mu: float = 0.035, discount_factor: float = 0.8):
        self.discount_factor = discount_factor
        self.arms_info = arms_info
        self.baseline_mu = baseline_mu
        self.rounds = rounds
        self.result = pd.DataFrame()

    def linear_to_log_params(self, mu_x, var_x):
        """
        Переводит (среднее, дисперсию) на линейной шкале (mu_x, var_x)
        в (mu_theta, sigma_theta^2) на лог-шкале логнормального распределения.
        Формулы:
          mu_theta = ln( mu_x^2 / sqrt(var_x + mu_x^2 ) )
          sigma_theta^2 = ln( var_x / mu_x^2 + 1 )
        """
        mu_theta = np.log((mu_x**2) / np.sqrt(var_x + mu_x**2))
        sigma_theta2 = np.log(var_x / (mu_x**2) + 1)

        return mu_theta, sigma_theta2

    def update_nig_oneobs(self, y_new, mu_0, lambda_0, alpha_0, beta_0):
        """
        Обновляет параметры NIG, учитывая ОДНО наблюдение y_new.
        Параметры:
          - y_new : наблюдение (лог(CPM))
          - (mu_0, lambda_0, alpha_0, beta_0): Текущие априорные/постериорные параметры
        Возвращает:
          (mu_n, lambda_n, alpha_n, beta_n) — новые постериорные параметры
        Формулы (n=1, см. Normal-Inverse-Gamma):
          alpha_n   = alpha_0 + n/2
          lambda_n  = lambda_0 + n
          mu_n      = (lambda_0*mu_0 + y_new) / (lambda_0 + n)
          beta_n    = beta_0 + ( lambda_0 / (2*(lambda_0+n)) ) * (y_new - mu_0)^2
        """
        alpha_n = alpha_0 + 0.5
        lambda_n = lambda_0 + 1.0
        mu_n = (lambda_0 * mu_0 + y_new) / (lambda_0 + 1.0)

        # Частная сумма квадратов для n=1
        beta_n = beta_0 + (lambda_0 / (2.0 * (lambda_0 + 1.0))) * (y_new - mu_0)**2

        return mu_n, lambda_n, alpha_n, beta_n

    def discounted_update_nig(self, y_new, mu_0, lambda_0, alpha_0, beta_0):
        """
        - Сначала "старим" (дисконтируем) текущие параметры,
          т. е. делаем их ближе к исходному (слабее).
        - Затем вызываем классический update_nig_oneobs для нового наблюдения.
        """

        # 1. "Старим" параметры
        lambda_0 = 1.0 + self.discount_factor * (lambda_0 - 1.0)
        alpha_0  = 2.0 + self.discount_factor * (alpha_0  - 2.0)
        beta_0   = 2.0 + self.discount_factor * (beta_0   - 2.0)

        mu_0 = self.baseline_mu + self.discount_factor * (mu_0 - self.baseline_mu)

        # 2. Теперь делаем обычное обновление одним наблюдением
        mu_n, lambda_n, alpha_n, beta_n = self.update_nig_oneobs(
            y_new, mu_0, lambda_0, alpha_0, beta_0
        )

        return mu_n, lambda_n, alpha_n, beta_n

    def sample_from_nig(self, mu_0, lambda_0, alpha_0, beta_0):
        """
        Сэмплирует (mu, sigma^2) из Normal-Inverse-Gamma:
          1) sigma^2 ~ InvGamma(alpha_0, beta_0)
          2) mu      ~ Normal(mu_0, sigma^2 / lambda_0)
        Возвращает (mu, sigma2).
        """
        # Сэмплируем sigma^2
        sigma2_sample = invgamma(a=alpha_0, scale=beta_0).rvs()
        # Сэмплируем mu
        mu_sample = norm(loc=mu_0, scale=np.sqrt(sigma2_sample / lambda_0)).rvs()

        return mu_sample, sigma2_sample

    def simulation(self):

        # Переводим (среднее, дисперсию) на линейной шкале (mu_x, var_x) в (mu_theta, sigma_theta^2) на лог-шкале
        true_mu_log = {}
        true_sigma2_log = {}
        for arm in self.arms_info.keys():
            mu_x = self.arms_info[arm]
            var_x = 0.01

            mu_theta, sigma_theta2 = self.linear_to_log_params(mu_x, var_x)
            true_mu_log[arm] = mu_theta
            true_sigma2_log[arm] = sigma_theta2

        # Инициализация NIG-параметров для каждой руки (на лог-шкале)
        nig_params = {}
        for arm in self.arms_info.keys():
            nig_params[arm] = {
                'mu': np.log(0.035),   # mu_0
                'lambda': 1.0,         # lambda_0 (чем меньше, тем "шире" доверие)
                'alpha': 2.0,          # alpha_0
                'beta': 2.0,           # beta_0
            }

        # Цикл Thompson Sampling с NIG + логнорм
        traffic_distribution = []
        bid_rates = []

        for t in tqdm(range(self.rounds)):
            # Шаг 1: Для каждой кампании сэмплируем (mu, sigma2) из NIG
            #        Затем получаем sampled_cpm = exp(mu)
            sampled_values = {}
            for arm in self.arms_info.keys():
                mu_s, sigma2_s = self.sample_from_nig(nig_params[arm]['mu'],
                                                      nig_params[arm]['lambda'],
                                                      nig_params[arm]['alpha'],
                                                      nig_params[arm]['beta'])
                # переводим из лог-шкалы
                sampled_values[arm] = np.exp(mu_s)

            # Шаг 2: Выбираем кампанию с максимальным сэмплированным значением
            chosen_arm = max(sampled_values, key=sampled_values.get)

            # Шаг 3: Генерируем фактическое значение, используя истинные параметры
            actual_value = np.random.lognormal(true_mu_log[chosen_arm], np.sqrt(true_sigma2_log[chosen_arm]))

            # Шаг 4: Обновляем NIG-параметры этой руки переводим фактическое значение в лог-шкалу => y_new
            y_new = np.log(actual_value)

            current = nig_params[chosen_arm]
            mu_new, lambda_new, alpha_new, beta_new = self.discounted_update_nig(
                y_new,
                current['mu'],
                current['lambda'],
                current['alpha'],
                current['beta']
            )
            nig_params[chosen_arm]['mu'] = mu_new
            nig_params[chosen_arm]['lambda'] = lambda_new
            nig_params[chosen_arm]['alpha'] = alpha_new
            nig_params[chosen_arm]['beta'] = beta_new

            # Сохраняем результат для статистики
            traffic_distribution.append(chosen_arm)
            bid_rates.append(sampled_values[chosen_arm])

        # Анализ результатов
        self.result = pd.DataFrame({
            'arm': traffic_distribution,
            'value': bid_rates
        })

        self.result = self.result.groupby('arm').agg(
            impressions=('value', 'count'),
            cpm=('value', 'mean')
        ).reset_index()

        print("Итоговая статистика по распределению трафика и ставкам:")
        display(self.result)