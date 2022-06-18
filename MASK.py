import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from marketing_attribution_models import MAM
import dice_ml
from dice_ml.utils import helpers

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from numpy.random import seed
import random

def main():

    
    st.set_page_config(page_title="MASK | Marketing Science Kit (Proto)",
                        page_icon=":bar_chart:" )
    
    st.sidebar.header("メニュー")
    marketing_kits = ["MASKについて", "マーケティングミックスモデル", "アトリビューションモデル", "反実仮想",
                        "Twitter分析", "ペルソナ分析", "離反顧客分析"]
    
    marketing_kit = st.sidebar.selectbox(
            'マーケティングサイエンスの手法を選択：',marketing_kits
        )

    df = pd.read_csv("id_level_journey.csv")

    if marketing_kit == "MASKについて":
        st.header("MASK(Marketing Science Kit):bar_chart:")
        st.subheader("<目的>")
        st.write("マーケティング投資を再現性を持って最適な金額にすることを実現するためのダッシュボードです。基本機能と発展機能に分かれます。")
        st.text("※各機能に搭載されているデータはダミーデータです")
        st.write("")
        
        st.write("--------")
        st.subheader("<基本機能>")
        st.write("・マーケティングミックスモデル")
        st.write("オンラインとオフラインの広告が成果指標に対してどの程度効いているか、また、いくらが最適な投資金額であるかを分析する手法です")    
        st.write("")

        st.write("・アトリビューションモデル")
        st.write("オンライン広告が成果指標に対してどの程度効いているかをID単位のデータを元に詳細に確認する手法です")
        st.write("")

        st.write("・反実仮想")
        st.write("コンバージョンしなかったユーザーをコンバージョンさせるには、どういう手段が有効であったかをシミュレーションとして提案する手法です")
        st.write("")

        st.write("-----------")
        st.subheader("<発展機能>")
        st.write("・Twitter分析")
        st.write("Twitterのつぶやきからプロダクト・サービス改善のヒントを得る手法です")
        st.write("")

        st.write("・ペルソナ分析")
        st.write("顧客理解の解像度をクラスタリング分析で上げるための手法です")
        st.write("")

        st.write("・離反顧客分析")
        st.write("離反顧客についての理解を深める・予測を行うための手法です")



    elif marketing_kit == "マーケティングミックスモデル":
        st.header("マーケティングミックスモデル:bar_chart:")
        #model file
        ping = '2_185_7'
        
        #training data / opt file
        training_data = "sales_data.csv" 
        optimized_file = "{}_reallocated.csv".format(ping)
        
        dolists = ["モデルの正確性確認", "広告チャネルの金額確認"]
        DoList = st.sidebar.selectbox(
            '確認したい事項を選択：',dolists
        )

        if DoList == "モデルの正確性確認":

            def gitmodelfit(pngfile):
                st.subheader('モデルの適合度')
                st.image("./{}.png".format(pngfile))
            
            pngfile = ping
            gitmodelfit(ping)


        elif DoList == "広告チャネルの金額確認":

            def visualization(training_data, optimzed_data ):
                st.subheader('広告最適化ダッシュボード')

                channels = ["シミュレート結果一覧",
                 "Email_Marketing_S","Facebook_S", "GoogleDisplay_S",'GoogleSearch_S', "Instagram_S", "Youtube_S"]
                selected_channel = st.sidebar.selectbox(
                '広告チャネルを選択：',channels)

                training_data_path = "{}".format(training_data)
                optimized_file = "{}".format(optimzed_data)

                
                #st.write(training_data_path)
                df_training = pd.read_csv(training_data_path, encoding="cp932")

                def investment(channel, file, data):
                    result_data = pd.read_csv(file, encoding="CP932")
                    selected = result_data.loc[result_data.loc[:,"channels"]==channel,:]
                    st.subheader("広告項目：{}".format(channel))
                    fig = plt.figure(figsize=(6,3))
                    plt.title("Spend for {} per day".format(channel))
                    plt.vlines(selected.loc[:, "initSpendUnit"], 0, 50, "0.3", linestyles='dashed', label="Current Ave")
                    plt.vlines(selected.loc[:, "optmSpendUnit"], 0, 50, "red", linestyles='dashed', label="Optm Ave")
                    st.write("＜1日あたりの{}消化金額サマリー＞".format(channel))
                    
                    names = ["現状投資額(Current Ave)","最適値投資額(Optm Ave)", "広告投資金額差分", "CPA改善差分"]
                    num_columns = len(names)
                    cols = st.columns(num_columns)
                    for name, col in zip(names, cols):
                        if name == names[0]:
                            try:
                                value = int(selected.loc[:, "initSpendUnit"].iloc[-1])
                                col.metric(label=name, value=f'{value:,} 円')
                            except IndexError:
                                st.write("＜データサンプル＞")
                                st.write("最初の期間の５件")
                                st.write(data.loc[ data.loc[:, channel]>0 ,["DATE", channel]].head())
                                st.write("最後の期間の５件")
                                st.write(data.loc[ data.loc[:, channel]>0 ,["DATE", channel]].tail())
                                value = data.loc[ data.loc[:, channel]>0 ,channel].mean()
                                col.metric(label=name, value=f'{value:,.3f} 円')
                                
                        if name == names[1]:
                            try:
                                value = int(selected.loc[:, "optmSpendUnit"].iloc[-1])
                                col.metric(label=name, value=f'{value:,} 円')
                            except:
                                value = 0
                                col.metric(label=name, value=f'{value:,.3f} 円')
                        if name == names[2]:
                            try:
                                value = int(selected.loc[:, "optmSpendUnit"].iloc[-1]) - int(selected.loc[:, "initSpendUnit"].iloc[-1])
                                col.metric(label=name, value=f'{value:,} 円')
                            except:
                                value = 0 - data.loc[ data.loc[:, channel]>0 ,channel].mean()
                                col.metric(label=name, value=f'{value:,.3f} 円')
                                st.write("※1 CPAへの寄与は確認されませんでした")
                                st.text("(現状投資金額に金額が入っている場合には回帰分析の結果の傾きは０の場合となります。金額が入っていない場合(nan)には投資をしていない広告配信戦略となります)")
                                st.write("※2 データ分析上は段階的に投資金額を減らしていくことが推奨されています")
                                st.text("(一律に0にするとCPAに影響が出る可能性があるため、十分に検討の上での意思決定が必要となります)")
                        if name == names[3]:
                            try:
                                value = selected.loc[:, "optmResponseUnitLift"].iloc[-1]
                                col.metric(label=name, value=f'{value:,.3f} 円')
                            except:
                                value = 0
                                col.metric(label=name, value=f'{value:,.3f} 円')         

                    
                    #st.write(data.loc[ data.loc[:, channel]>0 ,channel])
                    st.write('')
                    plt.hist(data.loc[ data.loc[:, channel]>0 ,channel], bins=6, color="0.8")
                    plt.legend()
                    st.write("＜グラフ：過去の平均消費金額と最適化＞")
                    st.text("（注）縦軸は出現回数、横軸は１日の投資金額")
                    st.pyplot(fig)
                    st.write('')
                    st.write('-----------------------------------------------------------------------')
                    
                if selected_channel == "シミュレート結果一覧":
                    st.image("{}_reallocated_hist.png".format(ping))
 

                if selected_channel == "Email_Marketing_S":
                    selected_channels = [ i for i in df_training.columns if "Email_Marketing" in i if "_S" in i]
                    #st.write(selected_channels)
                    for channel in selected_channels:
                        #st.write(channel)
                        investment(channel, optimized_file, df_training)      

                if selected_channel == "Facebook_S":
                    selected_channels = [ i for i in df_training.columns if "Facebook" in i if "_S" in i]
                    #st.write(selected_channels)
                    for channel in selected_channels:
                        #st.write(channel)
                        investment(channel, optimized_file, df_training)      

                if selected_channel == "GoogleDisplay_S":
                    selected_channels = [ i for i in df_training.columns if "GoogleDisplay" in i if "_S" in i]
                    #st.write(selected_channels)
                    for channel in selected_channels:
                        #st.write(channel)
                        investment(channel, optimized_file, df_training)      

                if selected_channel == "GoogleSearch_S":
                    selected_channels = [ i for i in df_training.columns if "GoogleSearch" in i if "_S" in i]
                    #st.write(selected_channels)
                    for channel in selected_channels:
                        #st.write(channel)
                        investment(channel, optimized_file, df_training)      

                if selected_channel == "Instagram_S":
                    selected_channels = [ i for i in df_training.columns if "Instagram" in i if "_S" in i]
                    #st.write(selected_channels)
                    for channel in selected_channels:
                        #st.write(channel)
                        investment(channel, optimized_file, df_training)      
 
                if selected_channel == "Youtube_S":
                    selected_channels = [ i for i in df_training.columns if "Youtube" in i if "_S" in i]
                    #st.write(selected_channels)
                    for channel in selected_channels:
                        #st.write(channel)
                        investment(channel, optimized_file, df_training)             

            visualization(training_data,optimized_file)
    

 
    elif marketing_kit == "アトリビューションモデル":
        st.header("アトリビューションモデル")
        
        
        attributions = MAM(df, group_channels=True,
        channels_colname = 'channels',
        journey_with_conv_colname= 'has_transaction',
        group_channels_by_id_list=['user_id'],
        group_timestamp_colname = 'visitStartTime',
        create_journey_id_based_on_conversion = True)
    
        attribution_markov = attributions.attribution_markov(transition_to_same_state=False)
        

        st.subheader("元データ")
        st.dataframe(df)
        st.write("")
        st.write("--------------")

        st.subheader("チャネル毎の貢献度")
        fig = plt.figure(figsize=(2,2))
        sns.heatmap(attribution_markov[3].round(3), cmap="YlGnBu", annot=True, linewidths=.5)
        st.pyplot(fig)
        st.write("")
        st.write("--------------")
        
        st.subheader("各種手法によるチャネル毎の貢献度")
        st.image('results.PNG')
        st.write("")
        st.write("--------------")
        
    
    
    elif marketing_kit == "反実仮想":
        np.random.seed(0)

        def generate_userprofile(nrows=50000):
            gender = ["male","female",]
            age = list(range(20, 80))
            rank =  ["A","B","C","D"]
            history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            rows = [
                [
                    random.choices(gender)[0],
                    random.choices(age)[0],
                    random.choices(rank)[0],
                    random.choices(history)[0],
                ]
                for _ in range(nrows)
            ]

            df = pd.DataFrame(
                data=rows,
                columns=["gender","age","rank","history"],
            )
            return df

        Organic = [10, 20, 30, 40, 50, 60 ,70, 80, 90, 100]
        Facebook = [2000, 1000, 4500]
        Google_Search = [4000, 5000, 10000]
        Instagram = [10000, 15000, 8000]
        Youtube=[15000, 20000, 8000]
        Google_Display = [3000, 4000, 4500]
        Email_Marketing= [3000, 2000, 1500]
        Direct = [3,4,5,6,7,8,9,10]
        AOV_month = {1: 120, 2:150, 3: 120, 4:150, 5:100, 6: 90, 7: 170, 8: 150, 9: 200, 10: 190, 11:190, 12:200}

        for i, value in enumerate(df["channels"]):  
            if value == "Organic":
                df.loc[i, 'spend'] = random.choices(Organic)[0]
            elif value == "Facebook":
                df.loc[i, 'spend'] = random.choices(Facebook)[0]
            elif value == "Google Search":
                df.loc[i, 'spend'] = random.choices(Google_Search)[0]
            elif value == "Instagram":
                df.loc[i, 'spend'] = random.choices(Instagram)[0]
            elif value == "Youtube":
                df.loc[i, 'spend'] = random.choices(Youtube)[0]
            elif value == "Google Display":
                df.loc[i, 'spend'] = random.choices(Google_Display)[0]
            elif value == "Email Marketing":
                df.loc[i, 'spend'] = random.choices(Email_Marketing)[0]
            elif value == "Direct":
                df.loc[i, 'spend'] = random.choices(Direct)[0]

        df_user = generate_userprofile()
        df_user_ = df_user.reset_index().copy()
        df_channel_user = pd.concat([df, df_user_],axis=1)
        df_dummy = df_channel_user[["channels", "gender", "rank"]]
        pd_df = pd.get_dummies(df_dummy)
        x = pd.concat([df_channel_user[["spend","month", "history"]], pd_df], axis=1)
        y = df_channel_user["has_transaction"]

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=123)

        model_logi = LogisticRegression(random_state=123, class_weight="balanced",  C=1,  penalty='l2', max_iter=100)
        model_logi.fit(train_x, train_y)
        st.dataframe(train_x)
        st.write(model_logi.score(test_x, test_y))
        y_pred = model_logi.predict(test_x)
        st.write("---------------")
        st.write(confusion_matrix(test_y, y_pred))

        d = dice_ml.Data(dataframe = pd.concat([test_x, test_y], axis=1),# データは、変数とアウトカムの両方が必要
                 continuous_features = ["spend"], #　連続変数の指定
                 outcome_name = "has_transaction")

        m = dice_ml.Model(model=model_logi,backend="sklearn")
        exp = dice_ml.Dice(d, m)

        st.write("---------------")
        st.write("")
        pre_counter = test_x.iloc[0:5, :] 
        st.write("分析対象のユーザー情報")
        st.dataframe(pre_counter)
        dice_exp = exp.generate_counterfactuals(pre_counter, # 反実仮想を生成したいもとデータ
                                                total_CFs=3, # 反実仮想の数
                                                desired_class = "opposite", # 目的とするクラスは反対方向へ、0、1などのクラスラベルでも良い 
                                            )
        st.write("-------------")
        st.write(dice_exp.visualize_as_dataframe(show_only_changes=True))# show_onlyで変数変化の差分のみを表示
        st.dataframe(dice_exp.visualize_as_dataframe(show_only_changes=True))
     

    elif marketing_kit == "Twitter分析":
        st.write("Twitter分析（プロダクト改善のための感情分析）の分析結果を確認したい場合には、追加でお問い合わせください")
        st.write("")
        st.write("＜イメージ＞")
        st.image("https://raw.githubusercontent.com/gucchi123/MarketingKits/main/WordCloud.jpg")
        st.text("ツイッターのデータを用いて、ポジネガ分析・WordCloud等のプロダクト改善のヒントをご提供します")
        st.write("")
        st.write("＜必要データ＞")
        st.write("Twitter API情報")
        
    
    
    elif marketing_kit == "ペルソナ分析":
        st.write("ペルソナ分析の分析結果を確認したい場合には、追加でお問い合わせください")
        st.write("")
        st.write("＜イメージ＞")
        st.image("https://raw.githubusercontent.com/gucchi123/MarketingKits/main/Free-Personas-Vector.png")
        st.write("")
        st.write("＜必要データ＞")
        st.write("ユーザーID単位でのユーザー登録情報・ユーザー行動データ・コンバージョン有無")
    

    elif marketing_kit == "離反顧客分析":
        st.write("離反分析の分析結果を確認したい場合には、追加でお問い合わせください")
        st.write("")
        st.write("＜イメージ＞")
        st.image("https://raw.githubusercontent.com/gucchi123/MarketingKits/main/churn.png")
        st.write("＜必要データ＞")
        st.write("ユーザーID単位でのユーザー登録情報・ユーザー行動データ・離反有無")

    #権利
    st.sidebar.write("Copyright © 2022 Makoto Mizuguchi. All Rights Reserved.")    


if __name__ == '__main__':
    main()