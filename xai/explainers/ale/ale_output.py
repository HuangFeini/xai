from explainers.ale import ale as ale
import os
import matplotlib.pyplot as plt

def ale_output(model,data,preserve=True):
    feature_list=data.columns.tolist()

    if preserve:
        for feature in feature_list:
            x_ale = ale.ale_plot(model=model, train_set=data, features=feature)

            ale_folder="ale_1"
            if os.path.exists(ale_folder):
                x_ale.to_csv(ale_folder + "\\" + feature+".csv")
            else:
                os.mkdir(ale_folder)
                x_ale.to_csv(ale_folder + "\\" + feature + ".csv")

def ale_plot_total(model,data):
    feature_list=data.columns.tolist()
    fig,ax=plt.subplots(2, int(len(feature_list) / 2))
    i=0
    for feature in feature_list:


        x_ale = ale.ale_plot(model=model, train_set=data, features=feature)
        #
        q=x_ale["quantiles"]
        v=x_ale["ALE"]
        if i < int(len(feature_list) / 2):
            x_ale.plot(x="quantiles",y="ALE",ax=ax[0, i])
        else:
            x_ale.plot(x="quantiles",y="ALE",ax=ax[1,i-int(len(feature_list) / 2)])

        i = i + 1
    plt.show()



