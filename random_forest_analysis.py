from fastai.tabular.all import (
    cont_cat_split,
    TabularPandas,
    FillMissing,
    Normalize,
    Categorify,
    TrainTestSplitter,
    RandomSplitter,
    IndexSplitter,
    range_of,
)

def random_forest_analysis(file, dep_var, reduction_method="accuracy", bound=0.0005, umap_op=True, n=250, split="stratify", index_col=None):
    import streamlit as st
    import time
    import datetime
    import pandas as pd 
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.tree import DecisionTreeClassifier
    #from dtreeviz.trees import *
    import streamlit.components.v1 as components
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

    import warnings
    warnings.filterwarnings('ignore')

    from PIL import Image 

    Image.MAX_IMAGE_PIXELS = 1000000000

    import os

    # from fpdf import FPDF
    pd.set_option('mode.chained_assignment', None)

    """Random Forest Data Exploration
    in: uploaded file, classificaiton column name
    out: the deemed most important features where redundant features have been removed and pdf summary report
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
    def data_reduction_loop(imp_offset, fi, xs, y, valid_xs, fi_drop_zeroes):
        """Finding Minimum Required Features Loop for Data Reduction Function
        in: value used to adjust important feature search (float), feature importance dataframetraining dataframe, training classification column, validation dataframe
        out: new offset value (float), reduced training set, validation training set, rf important model, predictions for validation set, features to keep (list)
        """

        imp_offset += 0.01
        
        #to_keep = fi[fi.imp > (threshold - imp_offset)].features
        to_keep = fi[fi.imp > fi_drop_zeroes["imp"].quantile(.99 - imp_offset)].features
        xs_imp = xs[to_keep]
        valid_xs_imp = valid_xs[to_keep]

        m_imp = rf(xs_imp, y, len(xs_imp))

        preds = m_imp.predict(valid_xs_imp)
        
        return imp_offset, xs_imp, valid_xs_imp, m_imp, preds, to_keep

    import numpy as np  
    def data_reduction(m, xs, y, valid_xs, valid_y, fi, reduction_method, bound):
        """Fuction that Returns Most Important Features that Keeps Required Accuracy and if Chosen Recall and Precision, also
        in: rf model, training dataframe, training classification column, validation dataframe, validation classification column, feature importance dataframe
        reduction method specification (str), bound (float)
        out: rf important model, reduced training set, kept feature names (list)
        """
        #print(fi["imp"].max())
        fi_drop_zeroes = fi.loc[fi['imp'] != 0]
        
        threshold = .99#0.005 #threshold = max(feature score)
        to_keep = fi[fi.imp>fi_drop_zeroes["imp"].quantile(.99)].features

        xs_imp = xs[to_keep]
        valid_xs_imp = valid_xs[to_keep]

        m_imp = rf(xs_imp, y, len(xs_imp))

        #Keep reactions that keep accuracy in acceptable bounds
        imp_offset = 0

        preds = m_imp.predict(valid_xs_imp)
        preds_full = m.predict(valid_xs)
        
        if reduction_method == "accuracy":
            while (m.score(valid_xs, valid_y) - m_imp.score(valid_xs_imp, valid_y)) > bound:
                if threshold - imp_offset == 0:
                    m_imp = rf(xs_imp, y, len(xs_imp))
                    break
                else:
                    imp_offset, xs_imp, valid_xs_imp, m_imp, preds, to_keep = data_reduction_loop(imp_offset, fi, xs, y, valid_xs, fi_drop_zeroes)
        else:
            while (precision_score(valid_y, preds_full, average='macro') - precision_score(valid_y, preds, average='macro')) > bound or (recall_score(valid_y, preds_full, average='macro') - recall_score(valid_y, preds, average='macro')) > bound or (m.score(valid_xs, valid_y) - m_imp.score(valid_xs_imp, valid_y)) > bound:
                if threshold - imp_offset == 0:
                    m_imp = rf(xs_imp, y, len(xs_imp))
                    break
                else:
                    imp_offset, xs_imp, valid_xs_imp, m_imp, preds, to_keep = data_reduction_loop(imp_offset, fi, xs, y, valid_xs, fi_drop_zeroes)

        return m_imp, xs_imp, valid_xs_imp, to_keep 
    
    def figsizecalc(n):
        """Function to Calculate Figure Size Based on "n"
        in: no. of classes (int)
        out: figure width (float)
        """
        leftmargin=0.5 #inches
        rightmargin=0.5 #inches
        categorysize=0.5 #inches
        figwidth = leftmargin + rightmargin + n*categorysize

        if figwidth < 0:
            figwidth = 15
        
        return figwidth

    # def save_image(img, img_name, type):
    #     """Saves matplotlib and plotly images to images folder and then to pdf
    #     in: matplotlib plt or plotly fig, name of image and specification if "plotly" or not
    #     """
    #     img_path = f"images/{img_name}.png"

    #     if type == "plotly":
    #         img.write_image(img_path)
    #     else:
    #         img.savefig(img_path)
    
    def cluster_columns(df, figwidth, feature_importance, font_size=8):
        """Clustering Function Based on Similarity Capable of Finding Redundant Features
        in: dataframe, width (float), feature importance dataframe
        out: features to drop (list)
        """
        if len(df) == 1: 
            pass
        else:
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

            #drop lowest importances
            to_drop = []
            for cluster in unique_data:
                importances = {}
                for feature in cluster:
                    importances[feature] = feature_importance[feature_importance["features"] == feature]["imp"].values[0]
                kept_feature = max(importances)
                    
                for feature in importances:
                    if feature != kept_feature:
                        to_drop.append(feature)

            plt.title("Similarity Dendrogram", fontsize=font_size_title-4) 
            try:   
                st.pyplot(fig)
                # save_image(plt, "dendrogram", "plt")
            except:
                st.error("Generation of dendrogram encountered errors")

            return to_drop
    
    def draw_umap(n_neighbors, min_dist, n_components, data, metric='euclidean'):
        """UMAP mapping 
        in: n_neighbours (int), min distnace (float), dimensions (int), dataframe
        out: dataframe
        """
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
    
    def output_time(time_in_seconds):
        """Time coversion for amount of seconds (probably function for this exists already)
        in: seconds (int)
        out: amount of time passed (str)
        """
        if time_in_seconds < 60:
            return f"{round(time_in_seconds)} seconds"
        elif (time_in_seconds >= 60) & (time_in_seconds < 60*60):
            minutes = int(time_in_seconds // 60)
            return f"{minutes} minutes {round(time_in_seconds - minutes * 60)} seconds"
        else:
            minutes = int(time_in_seconds // 60)
            hours = int(minutes // 60)
            return f"{hours} hours {minutes - hours * 60} minutes {round(time_in_seconds - minutes * 60)} seconds"
    
    def confusion_matrices_generation(which, dif1=0, dif2=0, dif3=0):
        """Generates confusion matrices and notes for the three times its called
        in: which instance (str), accuracy difference (float), precision difference (float), recall difference (float)
        """
        with st.spinner("Generating confusion matrices..."): 
            if which=="initial":
                i = 1
            elif which=="secondary":
                i = 2
            else:
                i = 3  

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
                # save_image(plt, f"non_normalized_conf_mat_{i}", "plt")
            with col2:
                ConfusionMatrixDisplay.from_predictions(valid_y, preds, 
                                                        display_labels=classnames, 
                                                        normalize='true', 
                                                        xticks_rotation=67.5, 
                                                        colorbar=False, 
                                                        cmap=plt.cm.Blues)
                plt.title("Normalized confusion matrix")
                st.pyplot(fig=plt)
                # save_image(plt, f"normalized_conf_mat_{i}", "plt")
            
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
        """Use as a fix for displaying dtree visualisation
        in: viz (dtree object), height (int), width (int)
        """
        dtree_html = f"<body>{viz.svg()}</body>"

        components.html(dtree_html, height=height, width=width)

    # def diagram_iterator(classnames, img_name, limit, start, spacing):
    #     """Adds grid layout structure to plots and adds pages if necessary to pdf
    #     in: class names (list), image set name (str), page limit (int), start point on page (int), spacing amount (int)
    #     out: end point on page (int)
    #     """
    #     for class_, i in zip(classnames, iter(range(len(classnames)))):
    #         if i*spacing + start >= limit and not i % 2:
    #             pdf.add_page()
    #             start = 0

    #         if i % 2:
    #             pdf.image(f"images/{img_name}_{class_}.png", 5, start+((i-1)/2)*spacing, width/2-10)
    #         else:
    #             act = i / 2 #actual no. that captures row iteration 
    #             pdf.image(f"images/{img_name}_{class_}.png", width/2, start+act*spacing, width/2-10)
    #             end = start+act*spacing

    #     return end

    ##### Script Start #####

    font_size_title=16
    
    filename, file_extension = os.path.splitext(file.name)
    if '.csv' == file_extension:
        sep=","
    else:
        sep='\t'

    with st.spinner("Reading uploaded file..."):
        # if dep_var == 'Phylum':
        #     file='gs://reaction_presence_test/reaction_abundance_combined_phylum.csv'
        if index_col != None:
            df = pd.read_csv(file, index_col=index_col, sep=sep)
        else:
            df = pd.read_csv(file, sep=sep)

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
        # dtm = DecisionTreeClassifier(max_leaf_nodes=4)
        # dtm.fit(xs, y)

        # if len(xs) > 500:
        #     samp_idx = np.random.permutation(len(y))[:500]
        # else:
        #     samp_idx = np.random.permutation(len(y))[:len(xs)]

        # viz = dtreeviz(dtm, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        # fontname='DejaVu Sans', scale=1.6, label_fontsize=10, class_names=classnames, title="Decision Tree")
        
        #training initial model on all features
        m = rf(xs, y, len(to.train))
        
        elapsed_time = time.perf_counter() - t

    # st_dtree(viz, 700, 600) 
    # with st.expander("See notes"):
    #     st.write("""
    #     This is a representation of a singular decision tree in the random forest used for this classification problem.
    #     The random forest is comprised of many decision trees. The model's decision classification is the majority vote of all the decision trees.
    #     """)

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
        # save_image(fig, "feature_importance", "plotly")

        with st.expander("See notes"):
            st.write(f"""
            Feature importance is calculated by the decrease in node impurity (a measure of how well the node splits the data), 
            weighted by the probability of reaching that node. 
            The most important features have the highest values, although a feature being on top in this diagram does not necessarily mean its *the* most important feature. 
            It is dependent on how the current model has arranged its trees. Redoing the analysis, you will see a different list. However, truly important features will 
            will continuously enter the top feature list, but most likely in different positions each time.
            """)
        
        m_imp, xs_imp, valid_xs_imp, to_keep = data_reduction(m, xs, y, valid_xs, valid_y, fi, reduction_method, bound)
        
        elapsed_time = time.perf_counter() - t

    st.success('Model generated with important features done! Time elapsed: {}'.format(output_time(elapsed_time)))

    preds = m_imp.predict(valid_xs_imp)
    precision_important = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall_important = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy_important = round(accuracy_score(valid_y, preds)*100, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of important features kept", f"{len(xs_imp.columns)}", f"{(len(xs_imp.columns)-len(xs.columns))} reduction in features", delta_color="off")
    with col2:
        st.metric("Reduction in dataset size", f"{round(((len(xs_imp.columns)-len(xs.columns))/len(xs.columns))*100, 2)}%")

    st.write(f"##### Model Metrics with the most important {len(xs_imp.columns)} features")
    col1, col2, col3 = st.columns(3)
    dif1 = round(accuracy_important-accuracy_initial, 2)
    dif2 = round(precision_important-precision_initial, 2)
    dif3 = round(recall_important-recall_initial, 2)
    col1.metric("Accuracy", str(accuracy_important) + "%", str(dif1) + "%")
    col2.metric("Precision", str(precision_important) + "%", str(dif2) + "%")
    col3.metric("Recall", str(recall_important) + "%", str(dif3) + "%")

    confusion_matrices_generation("secondary", dif1, dif2, dif3)
    with st.spinner("Checking for redundant features..."):

        #feature importance important features
        fi = rf_feat_importance(m_imp, xs_imp)
        
        #redundancy check
        figwidth = figsizecalc(len(xs_imp.columns) - 10)
        to_drop = cluster_columns(xs_imp, figwidth, fi)
        
        elapsed_time = time.perf_counter() - t
    with st.expander("See notes"):
        st.write(f"""
        Similar features are clustered together, merged early and far from the root; of these clusters only the feature with the highest importance is kept, the rest dropped. 
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

    precision_final = round(precision_score(valid_y, preds, average='macro')*100, 2)
    recall_final = round(recall_score(valid_y, preds, average='macro')*100, 2)
    accuracy_final = round(accuracy_score(valid_y, preds)*100, 2)

    st.metric("Final number of features", f"{len(xs_final.columns)}", f"{-1*((len(xs_final.columns)-len(xs_imp.columns)))} redundant features dropped", delta_color="off")
    st.write(f"##### Model Metrics with the final {len(xs_final.columns)} features")
    col1, col2, col3 = st.columns(3)
    dif1 = round(accuracy_final-accuracy_initial, 2)
    dif2 = round(precision_final-precision_initial, 2)
    dif3 = round(recall_final-recall_initial, 2)
    col1.metric("Accuracy", str(accuracy_final) + "%", str(dif1) + "%")
    col2.metric("Precision", str(precision_final) + "%", str(dif2) + "%")
    col3.metric("Recall", str(recall_final) + "%", str(dif3) + "%")

    confusion_matrices_generation("final", dif1, dif2, dif3)

    #feature importance final features
    fi = rf_feat_importance(m_final, xs_final)
    
    with st.spinner("Generating heatmap and histograms of top features..."):
        
        # if len(xs_final.columns) > 20:
        #     df_train = xs_final[:20].join(y, how="left")
        # else:
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
            # save_image(fig, f"heatmap_{class_}", "plotly")

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
            # save_image(plt, f"histogram_{class_}", "plt")

        with st.expander("See notes"):
            st.write(f"""
            Histograms of the distributions for each of the final features can be quite insightful into the models decision making process. 
            If there is clear separability in the data, one would expect there to also be clear visual differences in the histogram plots. If you see that
            the x-axis of the histograms does not correspond with the values in the uploaded file, that is becuase your data may have been normailized or that
            categorical variables have been encoded. 
            """)
        
        elapsed_time = time.perf_counter() - t
    st.success('Heatmap and histogram done! Time elapsed: {}'.format(output_time(elapsed_time)))

    if len(classnames) > 3:
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

            plt.title("LDA 2D", fontsize=font_size_title)
            st.pyplot(fig=plt, dpi=300)
            # save_image(plt, f"lda_2d", "plt")

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

            ax.legend(handles=sc.legend_elements(alpha=1)[0], labels=true_class_names, loc='upper left', bbox_to_anchor=(1.05, 1), prop={"size":10}) #, bbox_to_anchor=(1.0125, 1), loc=2
            
            plt.tight_layout()
            plt.title("LDA 3D", fontsize=font_size_title)
            st.pyplot(fig=plt, dpi=300)
            # save_image(plt, f"lda_3d", "plt")

            with st.expander("See notes"):
                st.write(f"""
                Linear Discriminant Analysis (LDA) can take the high dimensional space of our final {len(xs_final.columns)} feature dataset and 
                reduce the dimensionality to 3 dimensions. LDA places emphasis on minimising variance for groups and maximising distance between groups 
                resulting in the visualisations shown. The visualisations effectively shows us that the datapoints can be easily grouped if there is clear separability
                but also reveals the datapoints the model may have struggled with. 
                """)
            
            fig = px.scatter(LDA_df, x="l1", y="l2", color=dep_var+" label", title="Interactive LDA 2D")
            fig.update_traces(marker=dict(line=dict(width=0.8,
                                        color='White')),
                    selector=dict(mode='markers'))
            st.plotly_chart(fig)

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

            #reconstructing initial dataframe (needed as dataframe has been processed with fastai)
            df_valid = valid_xs_final.join(valid_y, how="left")

            frames = [df_train, df_valid]
            df_combined = pd.concat(frames)

            df_final = df_combined.drop(columns=[dep_var])
            target_col = df_combined[dep_var]
            label_col = df[dep_var]

            cmap = ListedColormap(sns.color_palette("hls", len(classnames)).as_hex())
        
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

            plt.title("UMAP 2D: n_neighbours={}".format(n), fontsize=font_size_title+10)
            plt.legend(prop={"size":20})
            #plt.figure(dpi=300)
            st.pyplot(fig=plt, dpi=300)
            # save_image(plt, f"umap_2d", "plt")

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
            plt.title("UMAP 3D: n_neighbours={}".format(n), fontsize=font_size_title)

            st.pyplot(fig=plt, dpi=300)
            # save_image(plt, f"umap_3d", "plt")

            with st.expander("See notes"):
                st.write(f"""
                UMAP's capability is to group unlabelled data based on similarity, done on the reduced dataset these are the resulting visualisations. 
                If the dataset is inherently divisible based on the final features, you should be able to see clear separation in the visualisations. 
                """)

            fig = px.scatter(umap_df_2d, x="u1", y="u2", color=dep_var+" label", title="Interactive UMAP 2D")
            fig.update_traces(marker=dict(line=dict(width=0.8,
                                        color='White')),
                selector=dict(mode='markers'))
            st.plotly_chart(fig)

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
    # else:
    #     pass

    elapsed_time = time.perf_counter() - t

    now = datetime.now().strftime("%H:%M:%S")

    st.metric("End time", now)

    st.success('Random forest analysis done! Time elapsed: {}'.format(output_time(elapsed_time)))

    # with st.spinner("Generating PDF..."):   

    #     #pdf initialisation and generation
    #     width = 210
    #     height = 297

    #     pdf = FPDF()
    #     pdf.add_page()

    #     #Title
    #     pdf.set_font('Arial', '', 24)  
    #     pdf.write(5, f"Random Forest Analysis Summary Report")
    #     pdf.ln(10)
    #     pdf.set_font('Arial', '', 16)
    #     pdf.write(4, f"Number of classes: {len(classnames)}")
    #     pdf.ln(5)
    #     pdf.write(4, f"Shape of dataset: {df.shape}")
    #     pdf.ln(5)
    #     pdf.write(4, f"Training and test size: {len(xs), len(valid_y)}")
    #     pdf.ln(5)

    #     #First page
    #     pdf.write(4, f"Model Metrics with all {len(xs.columns)} features")
    #     pdf.ln(5)
    #     pdf.write(4, f"Accuracy, Precision, Recall: {accuracy_initial}%, {precision_initial}%, {recall_initial}%")
    #     pdf.image("images/normalized_conf_mat_1.png", 5, 60, width/2-10)
    #     pdf.image("images/non_normalized_conf_mat_1.png", width/2, 60, width/2-10)
    #     pdf.image("images/feature_importance.png", 5, 155, width, height/3+25)

    #     #Second page
    #     pdf.add_page()
    #     pdf.write(4, f"Model Metrics with the {len(xs_imp.columns)} most important features")
    #     pdf.ln(5)
    #     pdf.write(4, f"Accuracy, Precision, Recall: {accuracy_important}%, {precision_important}%, {recall_important}%")
    #     pdf.image("images/normalized_conf_mat_2.png", 5, 60, width/2-10)
    #     pdf.image("images/non_normalized_conf_mat_2.png", width/2, 60, width/2-10)
    #     pdf.image("images/dendrogram.png", 5, 155, width-40, height/3+25)

    #     #Third & following pages
    #     pdf.add_page()
    #     pdf.write(4, f"Model Metrics without redundant features") #{', '.join(to_drop)}
    #     pdf.ln(5)
    #     pdf.write(4, f"Accuracy, Precision, Recall: {accuracy_final}%, {precision_final}%, {recall_final}%")
    #     pdf.image("images/normalized_conf_mat_3.png", 5, 30, width/2-10)
    #     pdf.image("images/non_normalized_conf_mat_3.png", width/2, 30, width/2-10)

    #     end = diagram_iterator(classnames, "heatmap", 270, 120, 90)
    #     end = diagram_iterator(classnames, "histogram", 270, end+90, 90)

    #     #Final pages
    #     pdf.add_page()
    #     pdf.image("images/lda_2d.png", 5, 30, width-40, height/3+25)
    #     pdf.image("images/lda_3d.png", 5, 150, width-40, height/3+25)
    #     if umap_op:
    #         pdf.add_page()
    #         pdf.image("images/umap_2d.png", 5, 30, width-40, height/3+25)
    #         pdf.image("images/umap_3d.png", 5, 150, width-40, height/3+25)

    # elapsed_time = time.perf_counter() - t
    # st.success("PDF done! Time elapsed: {}".format(output_time(elapsed_time)))

    return to_keep