from input_class import Context ,Separate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



class Prepare():

    def __init__(self , inputmodel : Separate):

        self.sample = inputmodel  
        
    def BoxPlot(self):
        plt_box=self.sample.concat_df.boxplot(column = 'label')
        plt_box.plot()
        plt.show()

    def isnull(self):
        missing_values = self.sample.df.isna().sum()
        total_missing = missing_values.sum()
        # print(missing_values)
        # print(total_missing)
        return total_missing
        
     
    def Check_Label(self):
        
        if len(self.sample.ylabel['label'].unique()) > 2:  
            self.sample.label_type = 1
            self.Label_Reduction()
            
        else:
            self.sample.label_type = 0
    


    def Label_Reduction(self):
        
        
        self.sample.ylabel['label']=self.sample.ylabel['label'].astype(str)
        # get_dummies to create binary columns for each label
        binary_columns = pd.get_dummies(self.sample.ylabel['label'], prefix='label' , dtype=int)

        # Concatenate the binary columns with the original DataFrame
        self.sample.concat_df = pd.concat([self.sample.df, binary_columns], axis=1)

        inertias = []
        silhouette_scores = []  # Added for silhouette scores
        n_clusters_max = 15

        self.sample.concat_df.columns = self.sample.concat_df.columns.astype(str)

        for k in range(2, n_clusters_max + 1):  # Changed starting point to 2 for silhouette score
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.sample.concat_df)
            inertias.append(kmeans.inertia_)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(self.sample.concat_df, labels))

        # Plot the elbow curve
        # plt.figure(figsize=(12, 6))

        # plt.subplot(1, 2, 1)
        # plt.plot(range(2, n_clusters_max + 1), inertias, marker='o')
        # plt.xlabel('Number of Clusters (k)')
        # plt.ylabel('Inertia')
        # plt.title('Elbow Method for Optimal k')

        # # Plot the silhouette scores
        # plt.subplot(1, 2, 2)
        # plt.plot(range(2, n_clusters_max + 1), silhouette_scores, marker='o')
        # plt.xlabel('Number of Clusters (k)')
        # plt.ylabel('Silhouette Score')
        # plt.title('Silhouette Score for Optimal k')

        # plt.tight_layout()
        # plt.show()

        # Find the optimal k based on the silhouette score
        optimal_k_silhouette = range(2, n_clusters_max + 1)[silhouette_scores.index(max(silhouette_scores))]

        # print(f'Optimal K based on Silhouette Score: {optimal_k_silhouette}')
        self.sample.concat_df.drop(binary_columns.columns, axis=1, inplace=True)
        # Apply K-Means clustering with the optimal k
        kmeans_optimal = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
        self.sample.concat_df['cluster_label'] = kmeans_optimal.fit_predict(self.sample.concat_df)
        binary_columns = pd.get_dummies(self.sample.concat_df['cluster_label'], prefix='cluster_label' , dtype=int)
       
        self.sample.concat_df = pd.concat([self.sample.concat_df, binary_columns], axis=1)
        label_columns = [col for col in self.sample.concat_df.columns if 'cluster_label_' in col]
        self.sample.ylabel = self.sample.concat_df[label_columns]

        # Display the DataFrame with cluster labels
        # self.sample.concat_df.to_csv('E:\\NegarUni\\apply\\NegarApply\\Project\\KNN\\OUT5-Y.csv', index=False)
    
     
        
    def LowVarianceFilter(self):
        
        variances = self.sample.df.var()
        # for column, variance in variances.items():
        #     print(f"Variance of {column}: {variance}")
        selected_columns = variances[variances > 0.06].index
        self.sample.df = self.sample.df[selected_columns]

    def Split(self):
        X = self.sample.df
        y = self.sample.ylabel
        self.sample.X_train, self.sample.X_test, self.sample.Y_train, self.sample.Y_test =train_test_split(X, y, test_size=0.2, random_state=12345)
        #train_test_split(X, y, test_size=0.2, random_state=42)
        
        
        print('X_train shape:', self.sample.X_train.shape)
        print('X_test shape:', self.sample.X_test.shape)
        print('Y_train shape:', self.sample.Y_train.shape)
        print('Y_test shape:',self.sample.Y_test.shape)
        
    def RandomForest(self):
            
        model = RandomForestRegressor(random_state=42, max_depth=10)
        model.fit(self.sample.X_train,self.sample.Y_train)
        features = self.sample.df.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[:370]  # top 370 features
        # plt.title('Feature Importances')
        # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        # plt.yticks(range(len(indices)), [features[i] for i in indices])
        # plt.xlabel('Relative Importance')
        # plt.show()
        # List of selected feature names based on importance
        selected_feature_names = [features[i] for i in indices]
        self.sample.df =self.sample.df.loc[:, selected_feature_names]
        
    
    

    def Standard(self):
        scaler = StandardScaler()
        self.sample.concat_df = pd.DataFrame(scaler.fit_transform(self.sample.df), columns=self.sample.df.columns)
        self.sample.X_train = pd.DataFrame(scaler.fit_transform(self.sample.X_train), columns=self.sample.X_train.columns)
        self.sample.X_test = pd.DataFrame(scaler.transform(self.sample.X_test), columns=self.sample.X_test.columns)
        
    def Optimal_Component(self):
        
        n_features = self.sample.df.shape[1]  
        n_components_range = range(1, n_features + 1)

        scores = []

        for n_components in n_components_range:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(self.sample.df)
            clf = RandomForestClassifier()  
            score = np.mean(cross_val_score(clf, X_pca, self.sample.ylabel, cv=5))
            scores.append(score)

        # Find the index of the maximum score
        best_index = np.argmax(scores)
        optimal_n_components = n_components_range[best_index]

        # Plot the cross-validation scores
        # plt.plot(n_components_range, scores, marker='o')
        # plt.xlabel('Number of Components')
        # plt.ylabel('Cross-Validation Score')
        # plt.title('Cross-Validation Score vs Number of Components')
        # plt.show()

        print(f"Optimal number of components: {optimal_n_components}")
        return optimal_n_components
    
    def Dimention_Reduction_PCA(self):
        
        n_components =400     #self.Optimal_Component()
        pca = PCA(n_components = n_components)
        X_pca = pca.fit_transform(self.sample.df)
        # explain_vaiance_ratio = pca.explained_variance_ratio_
        # print("Explain variance : " , explain_vaiance_ratio)
        
        pca_df = pd.DataFrame( X_pca , columns=[f"PC{i+1}" for i in range(n_components)])
        # pca_df['label'] = self.sample.ylabel
        self.sample.df = pca_df
        print(len(pca_df.axes[1]))
        
     
    
    def get_obj(self):
        return self.sample
    
    def PCA(self):
        
        # Perform PCA for dimension reduction
        n_components = 2
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(self.sample.X_train)
        X_test_pca = pca.transform(self.sample.X_test)