import streamlit as st
from random_forest_analysis import random_forest_analysis

##streamlit start##

st.write("""
# Random Forest Analysis
## *Data analysis and reduction through random forests*
""")

with st.expander("See explanation"):
    st.write("""
    Machine learning is an effective tool to classify but also infer feature (an individual measurable property of the dataset) importance. 
    Knowledge of feature importance leads to reduced data size and a clearer picture of what features are inherent to and defining of specific classifications.

    Recent studies have shown that structured data can be best modeled by ensembles of decision trees such as random forest or gradient boosting machines. 
    Random forests train faster than gradient boosting machines and requires significantly less hyperparameter tuning. 
    Although gradient boosting machines yield slightly better results, they are better used as a tool for specific probing. 
    Random forests provide a great “one size fits all” approach.

    To discover the minimum required data to effectively classify data, feature importance needs to be gauged. 
    In a random forest, feature importance is calculated by the decrease in node impurity which is a measure of how well the node splits the data, weighted by the probability of reaching that node. 
    The probability of reaching a node is calculated by the number of data entries that reach that specific node, divided by the total number of data entries. 
    The most important features have the highest values. If the most important features are known the dataset can be reduced to these features and maintain levels accuracy, recall and precision that remain close to the initial model’s levels which is trained on the entire dataset. 

    To do this new models must be continuously trained on a growing number of important features until these metrics are within acceptable bounds of the metric standards of the initial model. 
    This is where random forest’s fast training time is of particular use and other machine learning models become obsolete. 

    Once an optimum amount of important features are established, any potential redundancy of these features can be found by determining similarity. 
    This is done by calculating the Spearman’s Rank-Order Correlation for the features. All but one of the closely related features can be removed with minimal impact to accuracy, thus, reducing the dataset size furthermore.
    The resulting reduction in the size of the dataset can be seen as reduction in the dimensionality of the dataset and now more easily lends itself to visualization techniques.
    Linear Discriminant Analysis is used to find an optimum representation of the reduced dataset in three dimensional space. 
    UMAP is used to visually show if the dataset has inherent divisibility.
    """)

with st.expander("Sidebar documentation"):
    st.write("""
    - *Reduction Method*: During the random forest continuous retraining phase, the stopping point comes when either the accuracy is within
    bound of the initially trained model's accuracy, or if accuracy, recall and precision are all within this bound of their respective 
    initially trained model's counterpart. 
    What is recommended is that if your data contains low amount of training samples for a specific class, then have the *reduction method* set to just accuracy
    is preferable, as recall and precision are most likely to be not high in this scenario. If this is not the case setting the *reduction method* to
    accuracy, precision and recall is then preferable. 

    - *Bound*: This is the *bound* described in *reduction method* above. 

    - *UMAP*: "Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE"
    \- https://umap-learn.readthedocs.io/en/latest/.
    It is optional as depending on the size of the data can take considerably long time to generate. 

    - *n_neighbours (UMAP)*: This setting is for *UMAP* and it is described that the greater the *n_neighbours* value the more is revealed about the
    global structure of the data, whereas, a small *n_neighbours* value reveals more the local structure.
    Default is **250** which is high enough in most cases to be considered to reveal the global structure. 

    - *Train-Test split*: This setting indicates how you would like the data to be divided into training and validation groups:
        - *random*: randomly split the data into an 80:20 divide. 
        - *stratified*: makes sure the 80:20 divide contains similar levels of each class across the divide. 
        - *index*: pass in a predefined index to test on a specific group.  
    """)
    
reduction_method = st.sidebar.selectbox("Reduction Method", ("accuracy", "accuracy, precision and recall"))
bound = st.sidebar.slider("Bound (10^-4)", min_value=1, max_value=10, value=5, step=1)
umap_op = st.sidebar.selectbox("UMAP", ("true", "false"))
n = st.sidebar.slider("n_neighbours (UMAP)", min_value=0, max_value=1000, value=250)
split = st.sidebar.selectbox("Train-Test split", ("random", "stratify", "index"))

file = st.file_uploader("", type=['csv'])

if not file:
    st.write("**Upload file to get started**")
else:
    # if len(file) > 1:
    #     st.warning(f"Maximum number of files reached. Only the first will be processed.")
    #     file = file[0]

    dep_var = st.text_input("Input dependent variable")

    if not dep_var:
        st.write("**Enter dependent variable to move on**")
    else:
        if split == "index":
            idx = st.text_input("Please put in list of index values")
            index_col = st.text_input("Please indicate the index column if there is one, if not leave blank")
            if index_col == "":
                index_col = None
            kept_features = random_forest_analysis(file, dep_var, reduction_method=reduction_method, umap_op=umap_op, n=n, bound=bound*10**-4, split=idx, index_col=index_col)
            st.write(f"Final features are: {kept_features}")
        else:
            kept_features = random_forest_analysis(file, dep_var, reduction_method=reduction_method, umap_op=umap_op, n=n, bound=bound*10**-4, split=split)
            st.write(f"Final features are: {kept_features}")


