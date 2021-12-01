import streamlit as st
import time
import datetime
import pandas as pd
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import umap.umap_ as umap

#import os

def random_forest_analysis(file, dep_var, reduction_method="accuracy", bound=0.0005, umap_op="true", n=250, idx=None):
    
    """
    Random Forest Data Exploration
    returns the deemed most important features where redundant features have been removed
    """
    
    #Random Forest Classifier
    def rf(xs, y, max_samples, n_estimators=40, max_features=0.5, min_samples_leaf=5, **kwargs):
        return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
            max_samples=max_samples, max_features=max_features,
            min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
    
    #Feature Importance
    def rf_feat_importance(m, df):
        return pd.DataFrame({'features':df.columns, 'imp':m.feature_importances_}
                           ).sort_values('imp', ascending=False)
    
    #Finding Minimum Required Features Loop for Data Reduction Function
    def data_reduction_loop(imp_offset, fi, xs, y, valid_xs):

        imp_offset += 0.0001
        to_keep = fi[fi.imp > (0.005 - imp_offset)].features
        xs_imp = xs[to_keep]
        valid_xs_imp = valid_xs[to_keep]

        m_imp = rf(xs_imp, y, len(xs_imp))

        preds = m_imp.predict(valid_xs_imp)
        
        return imp_offset, xs_imp, valid_xs_imp, m_imp, preds, to_keep
    
    #Fuction that Returns Most Important Features that Keeps Required Accuracy and if Chosen Recall and Precision, also  
    def data_reduction(m, xs, y, valid_xs, valid_y, fi, reduction_method, bound):
        to_keep = fi[fi.imp>0.005].features

        xs_imp = xs[to_keep]
        valid_xs_imp = valid_xs[to_keep]

        m_imp = rf(xs_imp, y, len(xs_imp))

        #Keep reactions that keep accuracy in acceptable bounds
        imp_offset = 0

        preds = m_imp.predict(valid_xs_imp)
        preds_full = m.predict(valid_xs)
        
        if reduction_method == "accuracy":
            while (m.score(valid_xs, valid_y) - m_imp.score(valid_xs_imp, valid_y)) > bound:
                if 0.005 - imp_offset == 0:
                    m_imp = rf(xs_imp, y, len(xs_imp))
                    break
                else:
                    imp_offset, xs_imp, valid_xs_imp, m_imp, preds, to_keep = data_reduction_loop(imp_offset, fi, xs, y, valid_xs)
        else:
            while (precision_score(valid_y, preds_full, average='macro') - precision_score(valid_y, preds, average='macro')) > bound or (recall_score(valid_y, preds_full, average='macro') - recall_score(valid_y, preds, average='macro')) > bound or (m.score(valid_xs, valid_y) - m_imp.score(valid_xs_imp, valid_y)) > bound:
                if 0.005 - imp_offset == 0:
                    m_imp = rf(xs_imp, y, len(xs_imp))
                    break
                else:
                    imp_offset, xs_imp, valid_xs_imp, m_imp, preds, to_keep = data_reduction_loop(imp_offset, fi, xs, y, valid_xs)

        return m_imp, xs_imp, valid_xs_imp, to_keep 
    
    #Function to Calculate Figure Size Based on "n"
    def figsizecalc(n):
        leftmargin=0.5 #inches
        rightmargin=0.5 #inches
        categorysize = 0.5 #inches
        figwidth = leftmargin + rightmargin + n*categorysize
        
        return figwidth
    
    #Clustering Function Based on Similarity Capable of Finding Redundant Features
    def cluster_columns(df, figwidth, font_size=8):
        corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
        corr_condensed = hc.distance.squareform(1-corr)
        z = hc.linkage(corr_condensed, method='average')
        fig = plt.figure(figsize=(figwidth, figwidth))
        hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)

        fl = fcluster(z, t=0.01, criterion='distance')

        col_list = list(df.columns)
        clusters = []
        for cluster_ind in fl:
            cluster = np.where(fl == cluster_ind)[0].tolist()
            names = []
            for ind in cluster:
                names.append(col_list[ind])

            if len(names) > 1:
                clusters.append(names)

        unique_data = list(map(list,set(map(tuple,clusters))))

        to_drop = []
        for lst in unique_data:
            for i in range(len(lst)):
                if i != 0:
                    to_drop.append(lst[i])
                else:
                    pass
            
        #plt.savefig(nextnonexistent(path_to_folder_other + "/top_features_dendrogram.png"))
        st.pyplot(fig)
        
        return to_drop
    
    #UMAP Fitting Function
    def draw_umap(n_neighbors, min_dist, n_components, data, metric='euclidean'):
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            init='spectral',
            verbose=False
        )
        u = fit.fit_transform(data)
        return u
    
    #Time Conversion
    def output_time(time_in_seconds):
        if time_in_seconds < 60:
            return str(round(time_in_seconds)) + " seconds"
        elif (time_in_seconds >= 60) & (time_in_seconds < 60*60):
            minutes = int(time_in_seconds // 60)
            return str(minutes) + " minutes " + str(round(time_in_seconds - minutes * 60)) + " seconds"
        else:
            minutes = int(time_in_seconds // 60)
            hours = int(minutes // 60)
            return str(hours) + " hours " + str(minutes - hours * 60) + " minutes " + str(round(time_in_seconds - minutes * 60)) + " seconds"
    
    ##### Script Start #####

    with st.spinner("Reading uploaded file..."):
        df = pd.read_csv(file)
    st.success("File read!")

    st.write(f"Shape of dataset: {df.shape}")

    classnames = list(df[dep_var].unique())
    classnames.sort()
        
    st.write(f"Number of classes: {len(classnames)}")
    
    now = datetime.now().strftime("%H:%M:%S")
    
    st.write(f"Start time: {now}")
    
    t = time.perf_counter() #total time recorded
    
    with st.spinner('Splitting data into training and validation sets...'):
        #fastai's processing into training and validation groups
        cont_list, cat_list = cont_cat_split(df, max_card=20, dep_var=dep_var)

        if idx == None:
            splits = TrainTestSplitter(test_size=0.2, stratify=df[dep_var])(range_of(df)) ##maybe make other splitters an option
        elif idx == "random":
            splits = RandomSplitter()(range_of(df))
        else:
            splits = IndexSplitter(idx)(range_of(df))
            
        to = TabularPandas(df, procs=[FillMissing, Normalize, Categorify],
                        cat_names = cat_list,
                        cont_names = cont_list,
                        y_names=dep_var,
                        splits=splits)
        
        elapsed_time = time.perf_counter() - t

    st.success('Train-test split done! Time elapsed: {}'.format(output_time(elapsed_time)))
    with st.spinner("Training model..."):
    
        xs,y = to.train.xs,to.train.y
        valid_xs,valid_y = to.valid.xs,to.valid.y
        
        #training initial model on all features
        m = rf(xs, y, len(to.train))
        
        elapsed_time = time.perf_counter() - t
    st.success('Initial Model training done! Time elapsed: {}'.format(output_time(elapsed_time)))
        
    preds = m.predict(valid_xs)

    st.write(f"###### Model Metrics with all {len(xs.columns)} features")
    precision = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy = round(accuracy_score(valid_y, preds)*100, 2)
    st.write(pd.DataFrame({'Score': [precision, recall, accuracy]}, index=["Precision", "Recall", "Accuracy"]))

    with st.spinner("Generating confusion matrices..."):    
        #confusion matrices 
        try:
            figwidth = figsizecalc(len(classnames))*4 + 8 #weight & bias needed as figures are grouped in one plot
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(figwidth, figwidth))
            for ms, ax in zip([["Confusion matrix, without normalization", None], ["Normalized confusion matrix", 'true']], axes.flatten()):
                disp = plot_confusion_matrix(m, valid_xs, valid_y,
                                            display_labels=classnames,
                                            cmap=plt.cm.Blues,
                                            ax=ax,
                                            normalize=ms[1],
                                            xticks_rotation=67.5,
                                            colorbar=False) 

                disp.ax_.set_title(ms[0])
        
            plt.tight_layout()
            st.pyplot(fig)
        except:
            pass
        
        elapsed_time = time.perf_counter() - t
    st.success('Confusion matrices generated! Time elapsed: {}'.format(output_time(elapsed_time)))
    with st.spinner("Generating model with important features..."):
        #feature importance all features
        fi = rf_feat_importance(m, xs)
        
        if len(xs.columns) < 30:
            top_feature_length = len(xs.columns)
        else:
            top_feature_length = 30
        
        #Plotting Function for Feature Importance
        fi[:top_feature_length].plot('features', 'imp', 'barh', figsize=(12,7), legend=False)
        
        title = "Top " + str(top_feature_length) + " Features"
        
        plt.title(title)
        st.pyplot(fig=plt)
        
        m_imp, xs_imp, valid_xs_imp, to_keep = data_reduction(m, xs, y, valid_xs, valid_y, fi, reduction_method, bound)
        
        elapsed_time = time.perf_counter() - t

    st.success('Model generated with important features done! Time elapsed: {}'.format(output_time(elapsed_time)))

    preds = m_imp.predict(valid_xs_imp)
    st.write(f"##### Reduction in features: {len(xs.columns)} " +  r"""$$\rightarrow$$""" + f" {len(xs_imp.columns)}")
    st.write("###### Model Metrics with important reactions only")
    precision = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy = round(accuracy_score(valid_y, preds)*100, 2)
    st.write(pd.DataFrame({'Score': [precision, recall, accuracy]}, index=["Precision", "Recall", "Accuracy"]))

    with st.spinner("Checking for redundant features..."):

        #feature importance important features
        fi = rf_feat_importance(m_imp, xs_imp)
        
        #redundancy check
        figwidth = figsizecalc(len(xs_imp.columns))
        to_drop = cluster_columns(xs_imp, figwidth)
        
        elapsed_time = time.perf_counter() - t
    st.success('Redundancy check done! Time elapsed: {}'.format(output_time(elapsed_time)))

    with st.spinner("Training final model..."):
        
        #remove redundant reactions from to_keep
        to_keep = list(to_keep)
        for el in to_drop:
            to_keep.remove(el)
        
        xs_final = xs_imp.drop(to_drop, axis=1)
        valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)
        
        #training on final features
        m_final = rf(xs_final, y, len(xs_final))
        
        elapsed_time = time.perf_counter() - t
    st.success('Final model done! Time elapsed: {}'.format(output_time(elapsed_time)))
    
    preds = m_final.predict(valid_xs_final)
    
    st.write(f"##### Reduction in features: {len(xs_imp.columns)} " +  r"""$$\rightarrow$$""" + f" {len(xs_final.columns)}")
    st.write("###### Model Metrics with Final Features")
    precision = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy = round(accuracy_score(valid_y, preds)*100, 2)
    st.write(pd.DataFrame({'Score': [precision, recall, accuracy]}, index=["Precision", "Recall", "Accuracy"]))

    #feature importance final features
    fi = rf_feat_importance(m_final, xs_final)
    
    with st.spinner("Generating heatmap and histograms of top features..."):

        #heatmap
        c_mat = xs_final.corr()
        figwidth = figsizecalc(len(xs_final.columns))
        fig = plt.figure(figsize=(figwidth, figwidth))
        
        title = "Final Features Heatmap"
        plt.title(title)
        sns.heatmap(c_mat, vmax = .8, square = True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
        #histogram  
        xs_final.hist(figsize = (figwidth, figwidth))
        
        plt.tight_layout()
        st.pyplot(fig=plt)
        
        elapsed_time = time.perf_counter() - t
    st.success('Heatmap and histogram done! Time elapsed: {}'.format(output_time(elapsed_time)))
    with st.spinner("Generating LDA plots..."):
    
        #reconstructing initial dataframe (needed as dataframe has been processed with fastai)
        df_train = xs_final.join(y, how="left")
        df_valid = valid_xs_final.join(valid_y, how="left")

        frames = [df_train, df_valid]
        df_combined = pd.concat(frames)

        df_final = df_combined.drop(columns=[dep_var])
        target_col = df_combined[dep_var]
        label_col = df[dep_var]
        
        #lda
        X_LDA = LDA(n_components=2).fit_transform(df_final, target_col)
        X_LDA_3d = LDA(n_components=3).fit_transform(df_final, target_col)
        
        LDA_df = pd.DataFrame({"l1": X_LDA[:, 0], "l2": X_LDA[:, 1], dep_var: target_col})
        LDA_df_3d = pd.DataFrame({"l1": X_LDA_3d[:, 0], "l2": X_LDA_3d[:, 1], "l3": X_LDA_3d[:, 2], dep_var: target_col})
        
        LDA_df = LDA_df.join(label_col, rsuffix=" label")
        LDA_df_3d = LDA_df_3d.join(label_col, rsuffix=" label")
        
        #lda 2d
        plt.figure(figsize=(15,15))
        sc = sns.scatterplot(
            x="l1", y="l2",
            hue=dep_var + " label",
            hue_order=classnames,
            palette=sns.color_palette("hls", len(classnames)),
            data=LDA_df,
            legend="full",
            alpha=1
        )

        sc.legend_.set_title(None)

        plt.title("LDA 2D")
        st.pyplot(fig=plt)
        
        #lda 3d
        cmap = ListedColormap(sns.color_palette("hls", len(classnames)).as_hex())

        ax = plt.figure(figsize=(15,15)).gca(projection='3d')

        sc = ax.scatter(
            xs=LDA_df_3d["l1"], 
            ys=LDA_df_3d["l2"], 
            zs=LDA_df_3d["l3"], 
            c=LDA_df_3d[dep_var], 
            cmap=cmap,
            alpha=1,
            edgecolors='w',
            linewidth=0.8
        )
        ax.set_xlabel('LDA-one')
        ax.set_ylabel('LDA-two')
        ax.set_zlabel('LDA-three')
        
        true_class_names = []
        for entry in sc.legend_elements()[1]:
            result = re.search(r"\{([A-Za-z0-9_]+)\}", entry)
            try:
                true_class_names.append(classnames[int(result.group(1))])
            except:
                pass

        leg = ax.legend(handles=sc.legend_elements(alpha=1)[0], labels=true_class_names, loc='upper left', bbox_to_anchor=(1.05, 1)) #, bbox_to_anchor=(1.0125, 1), loc=2
        
        plt.tight_layout()
        plt.title("LDA 3D")
        st.pyplot(fig=plt)
        
        elapsed_time = time.perf_counter() - t
    st.success('LDA done! Time elapsed: {}'.format(output_time(elapsed_time)))
    
    #umap
    if umap_op == "true":
        with st.spinner("Generating UMAP plots..."):
        
            #umap 2d
            umap_result_2d = draw_umap(n, 0.1, 2, df_final) #sparsity??
            umap_df_2d = pd.DataFrame({"u1": umap_result_2d[:,0], "u2": umap_result_2d[:,1], dep_var: target_col})
            umap_df_2d = umap_df_2d.join(label_col, rsuffix=" label")
            
            plt.figure(figsize=(15,15))
            sc = sns.scatterplot(
                x="u1", y="u2",
                hue=dep_var + " label",
                hue_order=classnames,
                palette=sns.color_palette("hls", len(classnames)),
                data=umap_df_2d,
                legend="full",
                alpha=1
            )

            sc.legend_.set_title(None)

            plt.title("UMAP 2D: n_neighbours={}".format(n))
            st.pyplot(fig=plt)
            
            #umap 3d
            umap_result = draw_umap(n, 0.1, 3, df_final)
            umap_df = pd.DataFrame({"u1": umap_result[:,0], "u2": umap_result[:,1], "u3": umap_result[:,2], dep_var: target_col})
            umap_df = umap_df.join(label_col, rsuffix=" label")
            
            ax = plt.figure(figsize=(15,15)).gca(projection='3d')
            sc = ax.scatter(
                xs=umap_df["u1"], 
                ys=umap_df["u2"], 
                zs=umap_df["u3"], 
                c=umap_df[dep_var], 
                cmap=cmap,
                alpha=1,
                edgecolors='w',
                linewidth=0.8
            )
            ax.set_xlabel('umap-one')
            ax.set_ylabel('umap-two')
            ax.set_zlabel('umap-three')

            true_class_names = []
            for entry in sc.legend_elements()[1]:
                result = re.search(r"\{([A-Za-z0-9_]+)\}", entry)
                try:
                    true_class_names.append(classnames[int(result.group(1))])
                except:
                    pass

            ax.legend(handles=sc.legend_elements(alpha=1)[0], labels=true_class_names, loc='upper left', bbox_to_anchor=(1.05, 1)) #, bbox_to_anchor=(1.0125, 1), loc=2
            
            plt.tight_layout()
            plt.title("UMAP 3D: n_neighbours={}".format(n))
            st.pyplot(fig=plt)

            elapsed_time = time.perf_counter() - t
        st.success('UMAP done! Time elapsed:: {}'.format(output_time(elapsed_time)))
    else:
        pass
    
    elapsed_time = time.perf_counter() - t
    st.success('Random forest analysis done! Time elapsed: {}'.format(output_time(elapsed_time)))
    
    now = datetime.now().strftime("%H:%M:%S")

    st.write(f"End time: {now}")
    
    return to_keep

##streamlit start##

st.title("Random Forest Analysis")

st.write("""
## Data reduction through random forests
""")

reduction_method = st.sidebar.selectbox("Reduction Method", ("accuracy", "accuracy, precision and recall"))
umap_op = st.sidebar.selectbox("UMAP", ("false", "true"))
n = st.sidebar.number_input("n_neighbours (UMAP)", min_value=0, value=250)

file = st.file_uploader("Upload file", type=['csv'])

if not file:
    st.write("**Upload file to get started**")
else:
    dep_var = st.text_input("Input dependent variable")
    if not dep_var:
        st.write("**Enter dependent variable to move on**")
    else:
        kept_features = random_forest_analysis(file, dep_var, reduction_method=reduction_method, umap_op=umap_op, n=n, idx="random")


