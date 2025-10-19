import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


DATA_PATH = "mental-heath-in-tech-2016_20161114.csv"
n_components = 40
n_clusters = 2 

def standardize_data(DATA_PATH):
    '''
    UPLOADING DATASET AND STANDARDIZES COLUMNS BEFORE STATISTICAL ANALYSIS
    '''
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'dataset not found')
    print('uploading dataset...')
    df = pd.read_csv(DATA_PATH, encoding="utf-8", quotechar='"')
    print("DATASET LOADED. Shape:", df.shape)


    # Clean 'What is your age?' column
    age_col = 'What is your age?'
    if age_col in df.columns: 
        print("\nCleaning age column...")
        df[age_col] = pd.to_numeric(df[age_col], errors='coerce') # Convert to numeric, coerce errors to NaN
        df.loc[(df[age_col] < 15) | (df[age_col] > 100), age_col] = np.nan # Remove unrealistic ages
        median_age = df[age_col].median() # Calculate median age   
        df.loc[df[age_col].isna(), age_col] = median_age# Fix the chained assignment warning by using loc
        print(f"Age statistics after cleaning:\n{df[age_col].describe()}")

    #CLEAN GENDER COLUMN
    male_variations = ['male', 'm', 'man', 'mail', 'mle' ,'cis male', 'male.' ,'male (cis)', 'sex is male', 'dude' ,'I\'m a man why didn\'t you make this a drop down question. you should of asked sex? And i would of ans...', 'm|']
    female_variations = ['female', 'f', 'woman', 'femal', 'femail', 'i identify as female.', 'female assigned at birth', 'cis female' ,'female/woman' , 'cisgender female' , 'female (props for making this a freeform field, though)']

    def normalize_gender(val):
        if not isinstance(val, str):
            return 'other'
        val_clean = val.lower().strip()  # lowercase and remove spaces
        if val_clean in male_variations:
            return 'male'
        elif val_clean in female_variations:
            return 'female'
        else:
            return 'other'

    df['gender'] = df['What is your gender?'].apply(normalize_gender)
    df.drop(columns=['What is your gender?'], inplace=True, errors="ignore")

    yes_values_variations = {'yes, always','i was aware of some',
                             'yes, i was aware of all of them',
                             'i was aware of some', 
                             'yes, they all did','yes',
                               'yes', 'yea', 'yeah', 'yup', 'y', 'of course', 
                               'absolutely', 'sure', 'definitely', 'affirmative', 
                               'yes, they do', 'yes, they have','some did'}
    
    no_values_variations = {'none did','No, i only became aware later',
                            'n/a (not currently aware)','no', 'nope', 'nop', 'n', 
                            'none', 'not that I am aware of', 'not aware of any', 'not sure', 
                            'unknown', 'no, none did'}
 
    #YES/NO NORMALIZATION
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    def normalize_yes_no(val):
        if not isinstance(val, str):
            return val
        val_clean = val.lower().strip()
        if val_clean in [v.lower() for v in yes_values_variations]:
            return 'Yes'
        elif val_clean in [v.lower() for v in no_values_variations]:
            return 'No'
        else:
            return val
        
    for col in cat_cols:
        try:
            df[col] = df[col].apply(normalize_yes_no) 
        except Exception as e:
            print(f"not normalized {col}: {e}")
    return df

def data_exploration(df_clean):
    df_clean = standardize_data(DATA_PATH)

    print("\nDataset shape:", df_clean.shape) # Basic info
    print("\nColumn types:")
    print(df_clean.dtypes.value_counts())

    # Missing values analysis
    missing = (df_clean.isna().mean() * 100).sort_values(ascending=False)
    print("\nTop missing (%) columns:")
    print(missing.head(10))
    high_missing = missing[missing > 20]
    if not high_missing.empty:
        print("\nColumns with >20% missing:")
        print(high_missing)
    # Count columns with more than 20% missing values
    high_missing_count = len(high_missing)
    print(f"\nNumber of columns with >20% missing values: {high_missing_count}")
    # Analyze yes/no columns
    yes_no_cols = df_clean.select_dtypes(include=['object']).apply(
        lambda x: x.dropna().isin(['Yes', 'No', 'Unknown']).all()
    )
    yes_no_cols = yes_no_cols[yes_no_cols].index

    print("\nAnalyzing Yes/No responses:")
    for col in yes_no_cols:
        counts = df_clean[col].value_counts()
        print(f"\n{col}:")
        print(counts)
        # Visualize Yes/No distributions
        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f'Distribution of responses: {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Categorical columns analysis
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\nFound {len(cat_cols)} categorical columns. Detailed analysis:")
    
    # Visualize top 10 categories
    for col in cat_cols:
        if df_clean[col].nunique() < 30:  # Only plot if not too many categories
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df_clean, y=col, order=df_clean[col].value_counts().index[:10])
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.show()

    # Numeric analysis
    num = df_clean.select_dtypes(include=[np.number])
    if not num.empty:
        print("\nDetailed numeric analysis:")
        print(num.describe().T)
        
    print("\nStatistical analysis complete.")

def descriptive_analysis(df_clean):
    '''
    performs DESCRIPTIVE ANALYSIS ON THE CLEANED DATASET
    '''
    print("\nAge Analysis:")
    age_col = 'What is your age?'

    # Basic statistics
    print("\nBasic Age Statistics:")
    print(df_clean[age_col].describe())

    # Age distribution visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_clean, x=age_col, bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    # Box plot for age
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_clean[age_col])
    plt.title('Age Distribution (Box Plot)')
    plt.xlabel('Age')
    plt.show()

    # Gender distribution analysis
    print("\nGender Analysis:")
    print("\nGender Distribution:")
    gender_counts = df_clean['gender'].value_counts()
    print(gender_counts)

    # Visualize gender distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_clean, x='gender')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

    # Age distribution by gender
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_clean, x='gender', y='What is your age?')
    plt.title('Age Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Age')
    plt.show()
    # Country analysis
    country_col = 'What country do you live in?'
    if country_col in df_clean.columns:
        print("\nCountry Distribution:")
        country_counts = df_clean[country_col].value_counts().head(10)
        print("\nTop 10 Countries:")
        print(country_counts)

        # Visualize top 10 countries
        plt.figure(figsize=(12, 6))
        sns.barplot(x=country_counts.values, y=country_counts.index)
        plt.title('Top 10 Countries Distribution')
        plt.xlabel('Count')
        plt.ylabel('Country')
        plt.tight_layout()
        plt.show()

        # Country distribution by mental health condition
        plt.figure(figsize=(12, 6))
        top_5_countries = df_clean[country_col].value_counts().head(5).index
        filtered_df = df_clean[df_clean[country_col].isin(top_5_countries)]
        sns.countplot(data=filtered_df, x=country_col, hue='Do you currently have a mental health disorder?')
        plt.title('Mental Health Conditions by Top 5 Countries')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    # Mental health condition analysis
    print("\nMental Health Condition Analysis:")
    mental_health_cols = [
        'Do you currently have a mental health disorder?',
        'Have you been diagnosed with a mental health condition by a medical professional?',
        'Have you ever sought treatment for a mental health issue from a mental health professional?'
    ]

    for col in mental_health_cols:
        if col in df_clean.columns:
            print(f"\n{col}:")
            counts = df_clean[col].value_counts()
            print(counts)
            
            # Visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts.index, y=counts.values)
            plt.title(f'Distribution: {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Cross analysis with gender
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_clean, x=col, hue='gender')
            plt.title(f'{col} by Gender')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # Workplace discussion comfort analysis
    comfort_cols = [
        'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
        'Would you feel comfortable discussing a mental health disorder with your coworkers?'
    ]

    for col in comfort_cols:
        if col in df_clean.columns:
            print(f"\n{col}:")
            comfort_counts = df_clean[col].value_counts()
            print(comfort_counts)

            # Visualization of comfort levels
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_clean, x=col, hue='gender')
            plt.title(f'Comfort Level Distribution by Gender: {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Cross analysis with mental health condition
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_clean, x=col, hue='Do you currently have a mental health disorder?')
            plt.title(f'Comfort Level by Mental Health Condition: {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # Employer support analysis
        support_cols = [
            'Does your employer provide mental health benefits as part of healthcare coverage?',
            'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?',
            'Does your employer offer resources to learn more about mental health concerns and options for seeking help?'
        ]

        for col in support_cols:
            if col in df_clean.columns:
                print(f"\nEmployer Support Analysis - {col}:")
                support_counts = df_clean[col].value_counts()
                print(support_counts)

                # Basic visualization
                plt.figure(figsize=(10, 6))
                sns.countplot(data=df_clean, x=col)
                plt.title(f'Distribution of Employer Support: {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

                # Cross analysis with mental health condition
                plt.figure(figsize=(12, 6))
                sns.countplot(data=df_clean, x=col, hue='Do you currently have a mental health disorder?')
                plt.title(f'Employer Support by Mental Health Condition: {col}')
                plt.xticks(rotation=45)
                plt.legend(title='Mental Health Condition', bbox_to_anchor=(1.05, 1))
                plt.tight_layout()
                plt.show()

def preprocessing(df_clean):
    '''
    PERFORMS PREPROCESSING ON THE CLEANED DATASET
    '''
    threshold = 0.3
    df = df_clean.loc[:, df_clean.isnull().mean() < threshold] # Drop columns with more than 20% missing
    df = df.drop_duplicates()  # Drop duplicated rows
    df = df.loc[:, df.nunique() > 1] # Drop columns with only one unique value (no variance)
    df.drop(columns=["Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?", 
                     "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:", 
                     "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?", 
                     'Why or why not?', 
                     'Why or why not?.1' , 
                     'What country do you live in?' , 
                     'Would you be willing to bring up a physical health issue with a potential employer in an interview?',
                     'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?',
                     'Do you think that discussing a physical health issue with previous employers would have negative consequences?'], inplace=True, errors="ignore")
    print("REMOVED DUPLICATES, FEW VARIANCE,NON-USEFULL DUPLICATES AND EMPTY COLUMNS. Shape:", df.shape)
    print(df.head(5) , 'Shape:' , df.shape)

    COLUMN_MAPPING = [
    {
        'new_name': 'mental_health_benefits',
        'current': 'Does your employer provide mental health benefits as part of healthcare coverage?',
        'previous': 'Have your previous employers provided mental health benefits?'
    },
    {
        'new_name': 'aware_of_options',
        'current': 'Do you know the options for mental health care available under your employer-provided coverage?',
        'previous': 'Were you aware of the options for mental health care provided by your previous employers?'
    },
    {
        'new_name': 'offers_resources',
        'current': 'Does your employer offer resources to learn more about mental health concerns and options for seeking help?',
        'previous': 'Did your previous employers provide resources to learn more about mental health issues and how to seek help?'
    },
    {
        'new_name': 'observed_neg_conseq',
        'current': 'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?',
        'previous': 'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?'
    },
    {
        'new_name': 'comfort_discuss_coworkers',
        'current': 'Would you feel comfortable discussing a mental health disorder with your coworkers?',
        'previous': 'Would you have been willing to discuss a mental health issue with your previous co-workers?'
    },
    {
        'new_name': 'comfort_discuss_supervisors',
        'current': 'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
        'previous': 'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?'
    }
    ]

    def consolidate_columns(df, new_col_name, col_current_root, col_previous_root):
        if col_current_root in df.columns and col_previous_root in df.columns:
            df[new_col_name] = df[col_current_root].combine_first(df[col_previous_root])# Combine current and previous responses, prioritizing current
            df.drop(columns=[col_current_root, col_previous_root], inplace=True, errors='ignore')# Drop the original columns after consolidation
        return df  

    for mapping in COLUMN_MAPPING:
        df = consolidate_columns(
            df=df,
            new_col_name=mapping['new_name'],
            col_current_root=mapping['current'],
            col_previous_root=mapping['previous'],
        )
        print(f"Consolidato: {mapping['new_name']}")

    print(f"Colonne dopo il consolidamento: {len(df.columns)}")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols: 
        df[col] = df[col].fillna(df[col].median())   #adds median to missing numerical values

    for col in cat_cols: 
        df[col] = df[col].fillna(df[col].mode()[0])  #adds mode to missing categorical values
    
    
    drop_cols = [c for c in cat_cols if df[c].value_counts(normalize=True).max() > 0.9]
    df.drop(columns=drop_cols, inplace=True)  #drops columns where one category dominates
    print(df.shape)
    print(df.head(5))

    #SEPRARATE DEMOGRAFIC FEATURES AND STANDARDIZATION
    # Define demographic/descriptive features that won't be used for clustering
    demographic_features = [
        'What is your age?',
        'gender',
        'How many employees does your company or organization have?',
        'Are you self-employed?',
        'Do you work remotely?',
        'What country do you work in?',
        'Which of the following best describes your work position?'
    ]
    
    # Separate demographic features
    demographic_df = df[demographic_features].copy()
    clustering_df = df.drop(columns=demographic_features)

    print("\nFeatures used for clustering:")
    print(clustering_df.columns.tolist())
    print("\nDemographic features (will be used for analysis):")
    print(demographic_features)


    clustering_cat_cols = [col for col in cat_cols if col not in demographic_features]
    # Create dummy variables for categorical columns in clustering data
    df_encoded = pd.get_dummies(clustering_df, columns=clustering_cat_cols, drop_first=True)
    df_encoded = df_encoded.astype(int)
    df_encoded = df_encoded.loc[:, df_encoded.sum() > 10]  # Remove rare categories
    print("Shape after encoding:", df_encoded.shape)


    #SCALING FOR PCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns)
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    df_scaled = selector.fit_transform(df_scaled)
    df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns)
    
    print('Final shape for clustering:', df_scaled.shape)

    return df_encoded, df_scaled, demographic_df

def dimentionality_reduction(df_scaled, n_components):
    '''
    apply PCA for dimentionality reduction
    '''
    print('PCA DIMENTIONALITY REDUCTION')

    pca = PCA(n_components=n_components, random_state=42)
    df_pca = pca.fit_transform(df_scaled)
    explained = np.sum(pca.explained_variance_ratio_)
    print(f"Explained Variance Ratio: {explained:.2%}")
    print(df_pca)
    print('printing explained variance ratio...')    
    print(pca.explained_variance_ratio_)

    #analysis of the best number of components
    pca_full = PCA()
    pca_full.fit(df_scaled)

    explained_full = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained_full)+1), explained_full, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance (Full)')
    plt.grid(True)
    plt.show()

    n_components_90 = np.argmax(explained_full >= 0.90) + 1
    n_components_95 = np.argmax(explained_full >= 0.95) + 1

    print(f"Components for 90% variance: {n_components_90}")
    print(f"Components for 95% variance: {n_components_95}")
    
    return df_pca, pca

def clustering(df_pca, n_clusters):

    '''
    Determine K via elbow method and silhouette analysis, then execute K-means clustering
    '''
    # Compute metrics for different numbers of clusters
    K_range = range(2, 8)
    inertia = []
    silhouette_scores = []
    
    print('ANALYZING OPTIMAL NUMBER OF CLUSTERS...')
    for k in K_range:
        # Fit KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(df_scaled)
        
        # Calculate metrics
        inertia.append(km.inertia_)
        silhouette_avg = silhouette_score(df_scaled, clusters)
        silhouette_scores.append(silhouette_avg)
        print(f"k={k}, Silhouette Score: {silhouette_avg:.3f}, Inertia: {km.inertia_:.0f}")
    
    # Plot elbow and silhouette analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Elbow plot
    ax1.plot(K_range, inertia, 'bo-')
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.grid(True)
    
    # Silhouette plot
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_title("Silhouette Score Analysis")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("example_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k from silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")
    print(f"Using specified number of clusters: {n_clusters}")
    
    # Perform final clustering
    print('\nPERFORMING K-MEANS CLUSTERING...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Calculate final silhouette score
    final_silhouette = silhouette_score(df_scaled, clusters)
    print(f"Final Silhouette Score: {final_silhouette:.3f}")
    
    # Print cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print("\nCluster sizes:")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} samples ({(c/len(clusters))*100:.1f}%)")
    
    return clusters, kmeans, df_scaled

def visual_clusters(df_pca, clusters):
    """
    Create comprehensive visualization of clusters using both 2D and 3D plots
    """
    # Create scatter plot
    scatter = plt.scatter(df_pca[:,0], df_pca[:,1], 
                         c=clusters, 
                         cmap='viridis',
                         s=100,
                         alpha=0.6,
                         edgecolor='white',
                         linewidth=0.5)
    
    # Add title and labels
    plt.title("Mental Health in Tech - Cluster Analysis", 
              fontsize=16, 
              pad=20)
    plt.xlabel("First Principal Component", fontsize=12)
    plt.ylabel("Second Principal Component", fontsize=12)
    cbar = plt.colorbar(scatter) # Add colorbar legend  #=^.^=
    cbar.set_label('Cluster', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3) # Add grid for better readability
    plt.tight_layout()
    plt.show()

def cluster_analysis(df_scaled, clusters, demographic_df):
    '''
    ANALYZE CLUSTERS BASED ON DEMOGRAPHIC FEATURES
    '''
    df_analysis = demographic_df.copy()
    df_analysis['Cluster'] = clusters

    # Analyze age distribution across clusters
    print("\nAge Analysis by Cluster:")
    for cluster in df_analysis['Cluster'].unique():
        print(f"\nCluster {cluster} Age Statistics:")
        print(df_analysis[df_analysis['Cluster'] == cluster]['What is your age?'].describe())
    
    # Gender distribution
    print("\nGender Distribution by Cluster:")
    gender_cluster = pd.crosstab(df_analysis['Cluster'], df_analysis['gender'], normalize='index') * 100
    print(gender_cluster)

    # Visualize gender distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_analysis, x='Cluster', hue='gender')
    plt.title('Gender Distribution by Cluster')
    plt.show()

    # Company size analysis
    print("\nCompany Size Distribution by Cluster:")
    size_cluster = pd.crosstab(df_analysis['Cluster'], 
                              df_analysis['How many employees does your company or organization have?'],
                              normalize='index') * 100
    print(size_cluster)

    # Remote work analysis
    print("\nRemote Work Distribution by Cluster:")
    remote_cluster = pd.crosstab(df_analysis['Cluster'], 
                                df_analysis['Do you work remotely?'],
                                normalize='index') * 100
    print(remote_cluster)

    # Work position analysis
    print("\nWork Position Distribution by Cluster:")
    position_cluster = pd.crosstab(df_analysis['Cluster'],
                                  df_analysis['Which of the following best describes your work position?'],
                                  normalize='index') * 100
    print(position_cluster)
   
    # Analyze feature importance using cluster centroids
    print("\nAnalyzing Top Features Distinguishing Clusters:")
    
    # Get cluster centers
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=df_scaled.columns)
    
    # Calculate absolute differences between clusters
    cluster_diffs = abs(cluster_centers.iloc[0] - cluster_centers.iloc[1])
    
    # Get top 5 distinguishing features
    top_features = cluster_diffs.nlargest(5)
    print("\nTop 5 Distinguishing Features:")
    for feature, diff in top_features.items():
        print(f"\nFeature: {feature}")
        print(f"Absolute difference between clusters: {diff:.3f}")
        print("Cluster means:")
        for i in range(n_clusters):
            print(f"Cluster {i}: {cluster_centers.iloc[i][feature]:.3f}")
            
    # Visualize top features
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_features.index, y=top_features.values)
    plt.title('Top 5 Features Distinguishing Between Clusters')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    


    
if __name__ == "__main__":
    df_clean = standardize_data(DATA_PATH)
    # data_exploration(df_clean)
    # descriptive_analysis(df_clean)
    df_encoded, df_scaled, demographic_df = preprocessing(df_clean)
    df_pca, pca= dimentionality_reduction(df_scaled, n_components)
    clusters, kmeans, df_scaled = clustering(df_pca, n_clusters)
    visual_clusters(df_pca, clusters)
    cluster_analysis(df_scaled, clusters, demographic_df)    
    

