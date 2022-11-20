import h5py
import pandas as pd
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

def get_trade_features(trades, ob_ts):
    counts = []
    volumes = []
    avgPrice = []
    vwap = []
    i = 0
    c_trades = 0
    v = 0
    a = 0
    vw = 0
    ts_trades = trades["TS"]
    amount_trades = trades["Amount"]
    price_trades = trades["Price"]
    for ts in tqdm(ob_ts[1:]):
        while True:
            if ts_trades[i] >= ts:
                break
            c_trades += 1
            v += amount_trades[i]
            a += price_trades[i]
            vw += amount_trades[i] * price_trades[i]
            i += 1
        if c_trades == 0:
            avgPrice.append(0)
            vwap.append(vwap[-1])
        else:
            avgPrice.append(a/c_trades)
            vwap.append(vw/v)
        counts.append(c_trades)
        volumes.append(v)
        c_trades = 0
        v = 0
        a = 0
        vw = 0
    
    return counts, volumes, avgPrice, vwap

def prepare_ob_data(path):
    f = h5py.File(path, 'r')
    ob = f["OB"]
    trades = f["Trades"]

    print("Start data preparing...")
    counts, volumes, avgPrice, vwap = get_trade_features(trades, ob["TS"])

    df_features = pd.DataFrame(data={ 
                           "spread": ob["Ask"][:,0] - ob["Bid"][:,0],
                           "spread_rate": np.concatenate(([0], ((ob["Ask"][1:,0] - ob["Bid"][1:,0]) - (ob["Ask"][:-1,0] - ob["Bid"][:-1,0])))),
                           "vwap": np.concatenate(([0], [0], np.array(vwap[1:])-np.array(vwap[0:-1]))),
                           "rate_bid_volume1": np.concatenate(([0], ob["BidV"][1:,0] - ob["BidV"][:-1,0])),
                           "rate_ask_volume1": np.concatenate(([0], ob["AskV"][1:,0] - ob["AskV"][:-1,0])),
                           "rate_bid_volume2": np.concatenate(([0], ob["BidV"][1:,1] - ob["BidV"][:-1,1])),
                           "rate_ask_volume2": np.concatenate(([0], ob["AskV"][1:,1] - ob["AskV"][:-1,1])),
                           "price_ask_depth1": (ob["Ask"][:,1] - ob["Ask"][:,0]) * ob["AskV"][:,0],
                           "price_ask_depth2": (ob["Ask"][:,2] - ob["Ask"][:,1]) * ob["AskV"][:,1],
                           "price_bid_depth1": (ob["Bid"][:,1] - ob["Bid"][:,0]) * ob["BidV"][:,0],
                           "price_bid_depth2": (ob["Bid"][:,2] - ob["Bid"][:,1]) * ob["BidV"][:,1],
                           "volume_ask_depth30": sum([ob["AskV"][:,i] for i in range(30)]),
                           "volume_bid_depth30": sum([ob["BidV"][:,i] for i in range(30)]),
                           "bid_price_rate": np.concatenate(([0], ob["Bid"][1:,0] - ob["Bid"][:-1,0])),
                           "ask_price_rate": np.concatenate(([0], ob["Ask"][1:,0] - ob["Ask"][:-1,0])),
                           "bid_price_rate1": np.concatenate(([0], ob["Bid"][1:,1] - ob["Bid"][:-1,1])),
                           "ask_price_rate1": np.concatenate(([0], ob["Ask"][1:,1] - ob["Ask"][:-1,1])),
                           "bid_price_rate2": np.concatenate(([0], ob["Bid"][1:,2] - ob["Bid"][:-1,2])),
                           "ask_price_rate2": np.concatenate(([0], ob["Ask"][1:,2] - ob["Ask"][:-1,2])),
                           "trade_volumes": np.concatenate(([0], volumes)),
                           "trade_price_rate": np.concatenate(([0], [0], np.array(avgPrice[:-1]) -  np.array(avgPrice[1:]))),
                           "n_orders": np.concatenate(([0], counts)),
                           "TS": ob["TS"],
                           })

    df_features["volume_diff"] = df_features["volume_ask_depth30"] - df_features["volume_bid_depth30"]
    return df_features

def split_train_test(df_features, path_train_return, split=0.8):
    df_features = df_features.drop(columns=["TS"])
    f = h5py.File(path_train_return, 'r')
    dest = f["Return"]

    y = dest["Res"]
    split_idx = int(len(y) * split)

    X_train = df_features.head(split_idx)
    X_test = df_features.tail(len(y) - split_idx)
    Y_train = y[:split_idx]
    Y_test = y[split_idx:]

    return X_train, X_test, Y_train, Y_test

def fit_model(X_train, X_test, Y_train, Y_test):
    model = CatBoostRegressor(iterations=250,
                              learning_rate=0.07,
                              depth=10,
                              eval_metric='R2',
                              verbose=10,
                              loss_function='RMSE',
                              random_seed=1488)
    
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    print("R^2 =", r2_score(Y_test, preds))
    return model

def train(path_ob_trades, path_train_return, path_to_save):
    df = prepare_ob_data(path_ob_trades)
    X_train, X_test, Y_train, Y_test = split_train_test(df, path_train_return)
    model = fit_model(X_train, X_test, Y_train, Y_test)
    model.save_model(path_to_save+"model.cbm")

def forecast(path_ob_trades, path_train_return, model_path):
    model = CatBoostRegressor()
    model.load_model(model_path)
    df = prepare_ob_data(path_ob_trades)
    print(df.columns)
    ts = df["TS"]
    df = df.drop(columns=["TS"])
    p = model.predict(df)

    res = pd.DataFrame(data={"TS": ts, "Res": p})
    res.to_hdf('result2.h5', key='Return', mode='w')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help="Mode strart fit/forecast", default="fit", dest="mode")
    parser.add_argument("-f_exch", dest="path_ob_trades",
                        help="Data ob-trades need for train and forecast", default="data.h5")
    parser.add_argument("-f_return", dest="path_train_return", default="result.h5")
    parser.add_argument("-f_model", dest="path_model", default="model.cbm")
    parser.add_argument("-f_model_save", dest="path_model_to_save", default="")
    args = parser.parse_args()

    if args.mode == "fit":
        model = train(args.path_ob_trades, args.path_train_return, args.path_model_to_save)
    else:
        forecast(args.path_ob_trades, args.path_train_return, args.path_model)
