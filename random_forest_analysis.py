import streamlit as st
import time
import datetime
import pandas as pd
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import *
import streamlit.components.v1 as components
import graphviz as graphviz
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import fcluster
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import umap.umap_ as umap

import plotly.express as px

def random_forest_analysis(file, dep_var, reduction_method="accuracy", bound=0.0005, umap_op=True, n=250, split="stratify", index_col=None):
    
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

        plt.title("Similarity Dendrogram", fontsize=font_size_title-4)    
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
    
    def confusion_matrices_generation(which, dif1=0, dif2=0, dif3=0):
        with st.spinner("Generating confusion matrices..."):   

            #confusion matrices 
            col1, col2, = st.columns(2)
            with col1:
                ConfusionMatrixDisplay.from_predictions(valid_y, preds, 
                                                        display_labels=classnames, 
                                                        normalize=None, 
                                                        xticks_rotation=67.5, 
                                                        colorbar=False, 
                                                        cmap=plt.cm.Blues)
                plt.title("Confusion matrix, without normalization")
                st.pyplot(fig=plt)
            with col2:
                ConfusionMatrixDisplay.from_predictions(valid_y, preds, 
                                                        display_labels=classnames, 
                                                        normalize='true', 
                                                        xticks_rotation=67.5, 
                                                        colorbar=False, 
                                                        cmap=plt.cm.Blues)
                plt.title("Normalized confusion matrix")
                st.pyplot(fig=plt)
            
            elapsed_time = time.perf_counter() - t
        if which == "initial":
            with st.expander("See notes"):
                st.write(f"""
                Confusion matrices are a summary of prediction results on a classification problem. These matrices list the "predicted label" on the bottom and
                "true label" on the left. The accuracy, precision and recall results are all derived from the confusion matrix.

                *TP*, *TN*, *FP*, *FN* - True Positive, True Negative, False Positive, False Negative

                *Accuracy* - TP+TN/TP+FP+FN+TN

                *Precision* - TP/TP+FP

                *Recall* - TP/TP+FN
                """)
        else:
            with st.expander("See notes"):
                st.write(f"""
                    When compared to initial model's corresponding metric:
                    - Accuracy {"did not change" if dif1 == 0 else (f"decreased by {-1*dif1}%" if dif1 < 0 else f"increased by {dif1}%")}
                    - Precision {"did not change" if dif2 == 0 else (f"decreased by {-1*dif2}%" if dif2 < 0 else f"increased by {dif2}%")}
                    - Recall {"did not change" if dif3 == 0 else (f"decreased by {-1*dif3}%" if dif3 < 0 else f"increased by {dif3}%")}
                    """)
        st.success('Confusion matrices generated! Time elapsed: {}'.format(output_time(elapsed_time)))

    def st_dtree(viz, height=None, width=None):

        dtree_html = f"<body>{viz.svg()}</body>"

        components.html(dtree_html, height=height, width=width)

    ##### Script Start #####

    font_size_title=16

    with st.spinner("Reading uploaded file..."):
        if index_col != None:
            df = pd.read_csv(file, index_col=index_col)
        else:
            df = pd.read_csv(file)
    st.success("File read!")

    classnames = list(df[dep_var].unique())
    classnames.sort()
    
    now = datetime.now().strftime("%H:%M:%S")

    st.metric("Shape of dataset", f"{df.shape[0]} rows, {df.shape[1] - 1} features")
    st.metric("Number of classes", len(classnames))
    st.metric("Start time", now)
    
    t = time.perf_counter() #total time recorded
    
    with st.spinner('Splitting data into training and validation sets...'):
        #fastai's processing into training and validation groups
        cont_list, cat_list = cont_cat_split(df, max_card=20, dep_var=dep_var)

        if split == "stratify":
            splits = TrainTestSplitter(test_size=0.2, stratify=df[dep_var])(range_of(df)) ##maybe make other splitters an option
        elif split == "random":
            splits = RandomSplitter()(range_of(df))
        else:
            splits = IndexSplitter(split)(range_of(df))
            
        to = TabularPandas(df, procs=[FillMissing, Normalize, Categorify],
                        cat_names = cat_list,
                        cont_names = cont_list,
                        y_names=dep_var,
                        splits=splits)
        
        elapsed_time = time.perf_counter() - t

    st.success('Train-test split done! Time elapsed: {}'.format(output_time(elapsed_time)))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training set size", len(to.train.xs))
    with col2:
        st.metric("Validation set size", len(to.valid.xs))

    with st.spinner("Training model..."):
    
        xs,y = to.train.xs,to.train.y
        valid_xs,valid_y = to.valid.xs,to.valid.y

        #Decision Tree example
        dtm = DecisionTreeClassifier(max_leaf_nodes=4)
        dtm.fit(xs, y)

        if len(xs) > 500:
            samp_idx = np.random.permutation(len(y))[:500]
        else:
            samp_idx = np.random.permutation(len(y))[:len(xs)]

        viz = dtreeviz(dtm, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10, class_names=classnames, title="Decision Tree")
        
        #training initial model on all features
        m = rf(xs, y, len(to.train))
        
        elapsed_time = time.perf_counter() - t

    st_dtree(viz, 700, 600) 
    with st.expander("See notes"):
        st.write("""
        This is a representation of a singular decision tree in the random forest used for this classification problem.
        The random forest is comprised of many decision trees. The model's decision classification is the majority vote of all the decision trees.
        """)
    #st.image(image, use_column_width=True)
    st.success('Initial Model training done! Time elapsed: {}'.format(output_time(elapsed_time)))
        
    preds = m.predict(valid_xs)

    st.write(f"##### Model Metrics with all {len(xs.columns)} features")
    precision_initial = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall_initial = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy_initial = round(accuracy_score(valid_y, preds)*100, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", str(accuracy_initial) + "%")
    col2.metric("Precision", str(precision_initial) + "%")
    col3.metric("Recall", str(recall_initial) + "%")

    confusion_matrices_generation("initial")
    with st.spinner("Generating model with important features..."):
        #feature importance all features
        fi = rf_feat_importance(m, xs)
        
        if len(xs.columns) < 30:
            top_feature_length = len(xs.columns)
        else:
            top_feature_length = 30
        
        title = "Top " + str(top_feature_length) + " Features"
        fig = px.histogram(fi[:top_feature_length], x="imp", y="features")
        fig.update_layout(
            title=title,
            xaxis_title="Feature Importance",
            yaxis_title="Features",
        )
        st.plotly_chart(fig)

        with st.expander("See notes"):
            st.write(f"""
            feature importance is calculated by the decrease in node impurity (a measure of how well the node splits the data), 
            weighted by the probability of reaching that node. 
            The most important features have the highest values, although a feature being on top in this diagram does not necessarily mean its *the* most important feature. 
            It is dependent on how the current model has arranged its trees. Redoing the analysis, you will see a different list. However, truly important features will 
            will continuously enter the top feature list, but most likely in different positions each time.
            """)
        
        m_imp, xs_imp, valid_xs_imp, to_keep = data_reduction(m, xs, y, valid_xs, valid_y, fi, reduction_method, bound)
        
        elapsed_time = time.perf_counter() - t

    st.success('Model generated with important features done! Time elapsed: {}'.format(output_time(elapsed_time)))

    preds = m_imp.predict(valid_xs_imp)
    #st.write(f"##### Reduction in features: {len(xs.columns)} " +  r"""$$\rightarrow$$""" + f" {len(xs_imp.columns)}")
    #st.write(f"##### Model Metrics with the most important {len(xs_imp.columns)} features:")
    precision_important = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall_important = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy_important = round(accuracy_score(valid_y, preds)*100, 2)
    #st.write(pd.DataFrame({'Score': [precision, recall, accuracy]}, index=["Precision", "Recall", "Accuracy"]))
    st.metric("Number of important features kept", f"{len(xs_imp.columns)}", f"{(len(xs_imp.columns)-len(xs.columns))} reduction in features", delta_color="off")
    st.write(f"##### Model Metrics with the most important {len(xs_imp.columns)} features")
    col1, col2, col3 = st.columns(3)
    dif1 = round(accuracy_important-accuracy_initial, 2)
    dif2 = round(precision_important-precision_initial, 2)
    dif3 = round(recall_important-recall_initial, 2)
    col1.metric("Accuracy", str(accuracy_important) + "%", str(dif1) + "%")
    col2.metric("Precision", str(precision_important) + "%", str(dif2) + "%")
    col3.metric("Recall", str(recall_important) + "%", str(dif3) + "%")

    confusion_matrices_generation("non-initial", dif1, dif2, dif3)
    with st.spinner("Checking for redundant features..."):

        #feature importance important features
        fi = rf_feat_importance(m_imp, xs_imp)
        
        #redundancy check
        figwidth = figsizecalc(len(xs_imp.columns) - 10)
        to_drop = cluster_columns(xs_imp, figwidth)
        
        elapsed_time = time.perf_counter() - t
    with st.expander("See notes"):
        st.write(f"""
        Similar features are clustered together, merged early and far from the root; of these clusters all but one are kept, the rest dropped. 
        In this case features {', '.join(to_drop)} were chosen to be dropped. 
        """)
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
    
    #st.write(f"##### Reduction in features: {len(xs_imp.columns)} " +  r"""$$\rightarrow$$""" + f" {len(xs_final.columns)}")

    precision_final = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall_final = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy_final = round(accuracy_score(valid_y, preds)*100, 2)
    #st.write(pd.DataFrame({'Score': [precision, recall, accuracy]}, index=["Precision", "Recall", "Accuracy"]))

    st.metric("Final number of features", f"{len(xs_final.columns)}", f"{-1*((len(xs_final.columns)-len(xs_imp.columns)))} redundant features dropped", delta_color="off")
    st.write(f"##### Model Metrics with the final {len(xs_final.columns)} features")
    col1, col2, col3 = st.columns(3)
    dif1 = round(accuracy_final-accuracy_initial, 2)
    dif2 = round(precision_final-precision_initial, 2)
    dif3 = round(recall_final-recall_initial, 2)
    col1.metric("Accuracy", str(accuracy_final) + "%", str(dif1) + "%")
    col2.metric("Precision", str(precision_final) + "%", str(dif2) + "%")
    col3.metric("Recall", str(recall_final) + "%", str(dif3) + "%")

    confusion_matrices_generation("non-initial", dif1, dif2, dif3)
    #feature importance final features
    fi = rf_feat_importance(m_final, xs_final)
    
    with st.spinner("Generating heatmap and histograms of top features..."):
        
        df_train = xs_final.join(y, how="left")
        #heatmap
        col1, col2 = st.columns(2)
        for class_, i in zip(classnames, iter(range(len(classnames)))):
            df_train_class_select = df_train[df_train[dep_var] == classnames.index(class_)]

            c_mat = df_train_class_select.drop(columns=[dep_var]).corr()
            c_mat_df = pd.DataFrame(c_mat, index=xs_final.columns, columns=xs_final.columns).fillna(0)
            fig = px.imshow(c_mat_df, title=f"Final Features Heatmap ({class_})", zmin=-1, zmax=1)
            fig.update_layout(width=400, height=400)
            fig.update_xaxes(
            tickangle = 90)

            if i % 2 > 0:
                with col2:
                    st.plotly_chart(fig)
            else:
                with col1:
                    st.plotly_chart(fig)

        with st.expander("See notes"):
            st.write(f"""
            Heatmaps give insight into the relationship between features. Bright yellow indicating that a feature's measurable property is closely 
            correlated with the corresponding feature, thus why the downward diagonal is always bright yellow if the feature is present for a specific class.

            Dark Blue indicates that the opposite is true - that there exists a negative correlation between two corresponding features.
            """)

        #histogram 
        figwidth = figsizecalc(len(xs_final.columns))
        col1, col2 = st.columns(2)
        for class_, i in zip(classnames, iter(range(len(classnames)))):
            df_train_class_select = df_train[df_train[dep_var] == classnames.index(class_)]

            df_train_class_select.drop(columns=[dep_var]).hist(figsize = (figwidth, figwidth))

            plt.suptitle(f"Histogram for each Final Feature ({class_})", y=1.02, fontsize=font_size_title)
            plt.tight_layout()

            if i % 2:
                with col2:
                    st.pyplot(fig=plt)
            else:
                with col1:
                    st.pyplot(fig=plt)

        with st.expander("See notes"):
            st.write(f"""
            Histograms of the distributions for each of the final features can be quite insightful into the models decision making process. 
            If there is clear separability in the data, one would expect there to also be clear visual differences in the histogram plots. If you see that
            the x-axis of the histograms does not correspond with the values in the uploaded file, that is becuase your data may have been normailized or that
            categorical variables have been encoded. 
            """)
        
        elapsed_time = time.perf_counter() - t
    st.success('Heatmap and histogram done! Time elapsed: {}'.format(output_time(elapsed_time)))
    with st.spinner("Generating LDA plots..."):
    
        #reconstructing initial dataframe (needed as dataframe has been processed with fastai)
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

        #col1, col2, = st.columns(2)
        #with col1:
        plt.title("LDA 2D", fontsize=font_size_title)
        #plt.figure(dpi=300)
        st.pyplot(fig=plt, dpi=600)

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

        leg = ax.legend(handles=sc.legend_elements(alpha=1)[0], labels=true_class_names, loc='upper left', bbox_to_anchor=(1.05, 1), prop={"size":10}) #, bbox_to_anchor=(1.0125, 1), loc=2
        
        #with col2:
        plt.tight_layout()
        plt.title("LDA 3D", fontsize=font_size_title)
        #plt.figure(dpi=300)
        st.pyplot(fig=plt, dpi=600)

        with st.expander("See notes"):
            st.write(f"""
            Linear Discriminant Analysis (LDA) can take the high dimensional space of our final {len(xs_final.columns)} feature dataset and 
            reduce the dimensionality to 3 dimensions. LDA places emphasis on minimising variance for groups and maximising distance between groups 
            resulting in the visualisations shown. The visualisations effectively shows us that the datapoints can be easily grouped if there is clear separability
            but also reveals the datapoints the model may have struggled with. 
            """)
        
        #col1, col2, = st.columns(2)
        #with col1:
        fig = px.scatter(LDA_df, x="l1", y="l2", color=dep_var+" label", title="Interactive LDA 2D")
        fig.update_traces(marker=dict(line=dict(width=0.8,
                                    color='White')),
                selector=dict(mode='markers'))
        st.plotly_chart(fig)
        #with col2:
        fig = px.scatter_3d(LDA_df_3d, x="l1", y="l2", z="l3", color=dep_var+" label", title="Interactive LDA 3D")
        fig.update_traces(marker=dict(line=dict(width=0.8,
                                    color='White')),
                selector=dict(mode='markers'))
        st.plotly_chart(fig)

        with st.expander("See notes"):
            st.write(f"""
            Enlarge the interacitve plots for a better experience!
            """)
        
        elapsed_time = time.perf_counter() - t
    st.success('LDA done! Time elapsed: {}'.format(output_time(elapsed_time)))
    
    #umap
    if umap_op:
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
            #col1, col2, = st.columns(2)
            #with col1:
            plt.title("UMAP 2D: n_neighbours={}".format(n), fontsize=font_size_title+10)
            plt.legend(prop={"size":20})
            #plt.figure(dpi=300)
            st.pyplot(fig=plt, dpi=600)

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
            #with col2:
            plt.tight_layout()
            plt.title("UMAP 3D: n_neighbours={}".format(n), fontsize=font_size_title)
            #plt.figure(dpi=300)
            st.pyplot(fig=plt, dpi=600)

            with st.expander("See notes"):
                st.write(f"""
                UMAP’s capability is to group unlabelled data based on similarity, done on the reduced dataset these are the resulting visualisations. 
                If the dataset is inherently divisible based on the final features, you should be able to see clear separation in the visualisations. 
                """)

            #col1, col2, = st.columns(2)
            #with col1:
            fig = px.scatter(umap_df_2d, x="u1", y="u2", color=dep_var+" label", title="Interactive UMAP 2D")
            fig.update_traces(marker=dict(line=dict(width=0.8,
                                        color='White')),
                selector=dict(mode='markers'))
            st.plotly_chart(fig)
            #with col2:
            fig = px.scatter_3d(umap_df, x="u1", y="u2", z="u3", color=dep_var+" label", title="Interactive UMAP 3D")
            fig.update_traces(marker=dict(line=dict(width=0.8,
                                        color='White')),
                selector=dict(mode='markers'))
            st.plotly_chart(fig)

            with st.expander("See notes"):
                st.write(f"""
                Enlarge the interacitve plots for a better experience!
                """)

            elapsed_time = time.perf_counter() - t
        st.success('UMAP done! Time elapsed: {}'.format(output_time(elapsed_time)))
    else:
        pass
    
    elapsed_time = time.perf_counter() - t
    st.success('Random forest analysis done! Time elapsed: {}'.format(output_time(elapsed_time)))
    
    now = datetime.now().strftime("%H:%M:%S")

    st.metric("End time", now)
    
    return to_keep