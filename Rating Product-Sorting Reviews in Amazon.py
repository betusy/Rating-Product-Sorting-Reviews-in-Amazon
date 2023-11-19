## Rating Product & Sorting Reviews in Amazon

import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv('/Users/betulyilmaz/Documents/Miuul/Measurement Problems/Case Study 2-Rating Product Sorting Reviews Amazon/amazon_review.csv')
df.head()

# Rating Product
# Urunlerin duz ortalama rate
df['overall'].mean() # 4.5875

# Puanlama zamanina gore agirliklandiralim.
df.loc[df['day_diff'] <= df['day_diff'].quantile(0.25), 'overall'].mean() # 4.6957
df.loc[(df['day_diff'] > df['day_diff'].quantile(0.25)) & (df['day_diff'] <= df['day_diff'].quantile(0.50)), 'overall'].mean() # 4.6361
df.loc[(df['day_diff'] > df['day_diff'].quantile(0.50)) & (df['day_diff'] <= df['day_diff'].quantile(0.75)), 'overall'].mean() # 4.5716
df.loc[df['day_diff'] > df['day_diff'].quantile(0.75), 'overall'].mean() # 4.4462

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return df.loc[df['day_diff'] <= df['day_diff'].quantile(0.25), 'overall'].mean() * w1 / 100 + \
           df.loc[(df['day_diff'] > df['day_diff'].quantile(0.25)) & (df['day_diff'] <= df['day_diff'].quantile(0.50)), 'overall'].mean() * w2 / 100 + \
           df.loc[(df['day_diff'] > df['day_diff'].quantile(0.50)) & (df['day_diff'] <= df['day_diff'].quantile(0.75)), 'overall'].mean() * w3 / 100 + \
           df.loc[df['day_diff'] > df['day_diff'].quantile(0.75), 'overall'].mean() * w4 / 100

time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22) # 4.5955

# Sorting Reviews

# helpful_yes degiskeninden yola cikarak helpful_no degiskenini olusturuyoruz.
df['helpful_no'] = df['total_vote'] - df['helpful_yes']
df.head()

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_pos_neg_diff", ascending=False).head(20)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_average_rating", ascending=False).head(20)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head(20) # en iyi sonucu veriyor.

# 20 yorumu belirleyelim
df.sort_values("wilson_lower_bound", ascending=False).head(20)







