#!/usr/bin/env python
# coding: utf-8

### Script version of the Merlin notebook, useful for remote execution ###
### Refer to the notebook for a fully commented version, this is just a stripped down version ###


import os
import sys
import shutil
sys.path.insert(0, '../../')

import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv('../../upload.env')

EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME') # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID') # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY') # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY') # you received it in your e-mail
LIMIT = 0
from evaluation.EvalRSRunner import EvalRSRunner
from evaluation.EvalRSRunner import ChallengeDataset
from reclist.abstractions import RecModel

# start the timer
START_TIME = time.time()

dataset = ChallengeDataset(force_download=False)  # note, if YES, the dataset will be donwloaded again
runner = EvalRSRunner(
    dataset=dataset,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    participant_id=PARTICIPANT_ID,
    bucket_name=BUCKET_NAME,
    email=EMAIL
    )

import nvtabular as nvt
import nvtabular.ops as ops
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags
import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.dataset import BatchedDataset
from merlin.schema.tags import Tags
from merlin.models.utils import schema_utils
from merlin.models.tf.core.transformations import PopularityLogitsCorrection
import tensorflow as tf
from tensorflow.keras import regularizers


def get_item_frequencies(train_ds):   
    """Utility function that returns a TF tensor with the items (tracks) frequency"""
    schema = train_ds.schema
    # Gets the item ids cardinality
    item_id_feature = schema.select_by_tag(Tags.ITEM_ID)
    item_id_feature_name = item_id_feature.column_names[0]
    cardinalities = schema_utils.categorical_cardinalities(schema)
    item_id_cardinality = cardinalities[item_id_feature_name]

    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    item_frequencies_df = (
        train_ds.to_ddf()
        .groupby(item_id_feature_name)
        .size()
        .to_frame("freq")
        .compute()
    )
    assert len(item_frequencies_df) <= item_id_cardinality
    assert item_frequencies_df.index.max() < item_id_cardinality

    # Completing the missing item ids and filling freq with 0
    item_frequencies_df = item_frequencies_df.reindex(
        np.arange(0, item_id_cardinality)
    ).fillna(0)
    assert len(item_frequencies_df) == item_id_cardinality

    item_frequencies_df = item_frequencies_df.sort_index()
    item_frequencies_df["dummy"] = 1
    item_frequencies_df["expected_id"] = item_frequencies_df["dummy"].cumsum() - 1
    assert (
        item_frequencies_df.index == item_frequencies_df["expected_id"]
    ).all(), f"The item id feature ({item_id_feature_name}) should be contiguous from 0 to {item_id_cardinality-1}"

    item_frequencies = tf.convert_to_tensor(
        item_frequencies_df["freq"].values
    )

    return item_frequencies

class MyRetrievalModel(RecModel):
    
    def __init__(self, items_df: pd.DataFrame, users_df: pd.DataFrame, top_k: int = 100, 
                 predict_batch_size=1024, **kwargs):
        super(RecModel, self).__init__()
        self.items_df = items_df
        self.users_df = users_df
        self.top_k = top_k
        self.predict_batch_size = predict_batch_size
        self.hparams = kwargs
    
    def get_nvtabular_preproc_workflow(self):
        """Defines an NVTabular preprocessing workflow"""
        
        user_id_col = ["user_id"] 
        user_cat_cols = ["country", "gender"]
        user_age_col = ["age"]

        user_continuous_cols = [
            'novelty_artist_avg_month', 'novelty_artist_avg_6months', 'novelty_artist_avg_year',
            'mainstreaminess_avg_month', 'mainstreaminess_avg_6months', 'mainstreaminess_avg_year', 'mainstreaminess_global',
            'relative_le_per_weekday1', 'relative_le_per_weekday2', 'relative_le_per_weekday3',
             'relative_le_per_weekday4', 'relative_le_per_weekday5','relative_le_per_weekday6', 'relative_le_per_weekday7',
             'relative_le_per_hour0', 'relative_le_per_hour1','relative_le_per_hour2', 'relative_le_per_hour3',
             'relative_le_per_hour4', 'relative_le_per_hour5', 'relative_le_per_hour6', 'relative_le_per_hour7', 
             'relative_le_per_hour8', 'relative_le_per_hour9', 'relative_le_per_hour10', 'relative_le_per_hour11',
             'relative_le_per_hour12', 'relative_le_per_hour13', 'relative_le_per_hour14', 'relative_le_per_hour15',
             'relative_le_per_hour16', 'relative_le_per_hour17','relative_le_per_hour18', 'relative_le_per_hour19',
             'relative_le_per_hour20', 'relative_le_per_hour21', 'relative_le_per_hour22', 'relative_le_per_hour23', 
        ]

        user_counts_cols = ['playcount', 'cnt_listeningevents', 'cnt_distinct_tracks', 
                                'cnt_distinct_artists', 'cnt_listeningevents_per_week']

        item_id = ["track_id"]
        item_features_cols = ["artist_id", "album_id"] 


        user_id = user_id_col >> ops.Categorify() >> ops.TagAsUserID()
        user_feat_cat = user_cat_cols >> ops.Categorify() >> ops.TagAsUserFeatures()
        age_boundaries = list(np.arange(0,100,5))
        user_age = user_age_col >> ops.FillMissing(0) >> ops.Bucketize(age_boundaries) >> ops.Categorify() >> ops.TagAsUserFeatures()
        user_feat_cont = user_continuous_cols >> ops.FillMedian() >> ops.Normalize() >> ops.TagAsUserFeatures()
        user_feat_count = user_counts_cols >> ops.Clip(min_value=1) >> ops.FillMedian() >> ops.LogOp() >> ops.Normalize() >> ops.TagAsUserFeatures()
        user_features = user_id + user_feat_cat + user_age + user_feat_cont + user_feat_count

        item_id = item_id >> ops.Categorify() >> ops.TagAsItemID()
        item_cat_feat = item_features_cols >> ops.Categorify() >> ops.TagAsItemFeatures()
        item_features = item_id + item_cat_feat

        outputs = user_features + item_features
        workflow = nvt.Workflow(outputs)
        return workflow
    
    def preprocess_dataset(self, events_df):
        """Preprocess the dataset using an NVTabular workflow"""        
        nvt_dataset = nvt.Dataset(events_df)
        
        CATEG_MAPPING_FOLDER = 'categories/'
        shutil.rmtree(CATEG_MAPPING_FOLDER, ignore_errors=True)
        
        nvt_workflow = self.get_nvtabular_preproc_workflow()
        nvt_workflow.fit(nvt_dataset)
        schema = nvt_workflow.output_schema
        
        # Loads mapping of categ features after the workflow is fit
        self.user_ids_mapping_df = pd.read_parquet(CATEG_MAPPING_FOLDER+'unique.user_id.parquet')[['user_id']]
        self.track_ids_mapping_df = pd.read_parquet(CATEG_MAPPING_FOLDER+'unique.track_id.parquet')[['track_id']]
        
        transformed_dataset = nvt_workflow.transform(nvt_dataset)        
        transformed_df = transformed_dataset.persist().repartition(npartitions=10)  
        
        return transformed_df, schema
    
        
    def get_item_retrieval_task(self):        
        """Defines the item retrieval task to be used by the retrieval models"""
        items_frequencies = get_item_frequencies(self.train_dataset)
        
        post_logits = None
        reg_factor = self.hparams['logq_correction_factor']
        if reg_factor > 0.0:
            post_logits = PopularityLogitsCorrection(
                items_frequencies, reg_factor=reg_factor, schema=self.schema
            )
        
        item_retrieval_task = mm.ItemRetrievalTask(self.schema, 
                                        logits_temperature=self.hparams['logits_temperature'],
                                         post_logits=post_logits)
        
        return item_retrieval_task

    
    def get_model(self):
        """Defines the model architecture. Needs to be overridden by the child class"""
        raise NotImplementedError()
    
    def compile_model(self, model):
        """Compiles the Keras model setting the metrics, loss, learning rate and optimizer"""
        metrics = [mm.TopKMetricsAggregator(mm.RecallAt(20), mm.MRRAt(20))]
        
        lerning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            self.hparams['lr'],
            decay_steps=self.hparams['lr_decay_steps'],
            decay_rate=self.hparams['lr_decay_rate'],
            staircase=True,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lerning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=self.hparams['label_smoothing'],
        )
        
        model.compile(optimizer, loss=loss, metrics=metrics, run_eagerly=False)
        
    def get_items_topk_recommender_model(
        self,
        train_dataset: Dataset, 
        schema, 
        model, 
        ):        
        """Converts a retrieval model into a Top-k recommender model, which
        takes only user features as input, generates the user representations 
        (e.g. by taking the user embedding for MF or using the user tower to generate it)
        and uses scores all cached item representations to return the most similar items 
        (P.s. This procedure would be done by an ANN engine in production)"""
        item_features = schema.select_by_tag(Tags.ITEM).column_names
        item_dataset = train_dataset.to_ddf()[item_features].drop_duplicates(subset=['track_id'], keep='last').compute()
        item_dataset = Dataset(item_dataset)

        return model.to_top_k_recommender(item_dataset, k=self.top_k)
        
    def train_model(self, model, train_dataset):
        model.fit(
            train_dataset,
            epochs=self.hparams['epochs'],
            batch_size=self.hparams['train_batch_size'],
            #steps_per_epoch=5,
            shuffle=True,
            drop_last=True,
            train_metrics_steps=100,
        )
        

    def train(self, train_df: pd.DataFrame):
        """
        Implement here your training logic. Since our example method is a simple random model,
        we actually don't use any training data to build the model, but you should ;-)
        At the end of training, make sure the class contains a trained model you can use in the predict method.
        """
            
        print("Merging events and user features")
        events_df = train_df.merge(self.users_df, on='user_id', how='inner')

        print("Start preprocessing")
        transformed_df, schema = self.preprocess_dataset(events_df)
        train_dataset = Dataset(transformed_df, schema=schema)
            
        self.train_dataset = train_dataset
        self.schema = self.train_dataset.schema
        
        print("Building the model")
        model = self.get_model()
        self.compile_model(model)                
        
        print("Start training")
        self.train_model(model, train_dataset)        
        self.trained_model = model
        
        print("Preparing retrieval model for prediction")
        self.topk_rec_model = self.get_items_topk_recommender_model(train_dataset, schema, model)
        
        print("Caching users transformed features")
        self.users_schema = schema.select_by_tag(Tags.USER)
        user_features = self.users_schema.column_names
        self.users_transformed_df = train_dataset.to_ddf()[user_features].drop_duplicates(subset=['user_id'], keep='last').compute()
        # Adding the raw (original) user id to the dataframe
        self.users_transformed_df = self.users_transformed_df.merge(self.user_ids_mapping_df.rename({'user_id': 'raw_user_id'}, axis=1),
                                                                    left_on='user_id', right_index=True)

        print("Training completed!")         

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        """        
        This function takes as input all the users that we want to predict the top-k items for, and 
        returns all the predicted songs.

        While in this example is just a random generator, the same logic in your implementation 
        would allow for batch predictions of all the target data points.        
        """
            
        self.predict_user_ids = user_ids
        
        print("Start prediction")
        print("# users:",len(user_ids))
        test_users_df = user_ids.rename({'user_id': 'raw_user_id'}, axis=1).merge(self.users_transformed_df, 
                                                       on='raw_user_id', how='left')
        self.test_users_df = test_users_df
        test_users_found_df = test_users_df[~test_users_df[test_users_df.columns[1]].isna()]
        test_users_not_found_df = test_users_df[test_users_df[test_users_df.columns[1]].isna()]        

        test_users_dataset = Dataset(test_users_found_df, self.users_schema)
        test_batched_dataset = BatchedDataset(
            test_users_dataset, batch_size=self.predict_batch_size, shuffle=False, schema=self.users_schema
        )        
        
        print(f"Predicting Top-{self.top_k} items for test users")
        predictions = self.topk_rec_model.predict(test_batched_dataset)[1].astype(np.int32) 
        
        print(f"Converting user ids and predicted item ids to the original ids")
        predictions_converted = self.convert_prediction_item_ids(predictions)
        
        user_ids_found_converted = test_users_found_df['raw_user_id'].values.astype(np.int32)        
        user_ids_not_found_converted = test_users_not_found_df['raw_user_id'].values.astype(np.int32)
        
        # Merging raw user id with top-k predictions
        user_predictions_found_converted = np.concatenate((np.expand_dims(user_ids_found_converted, -1), predictions_converted), axis=1)
        user_predictions_not_found_converted = np.concatenate((np.expand_dims(user_ids_not_found_converted, -1), 
                                                               np.zeros((user_ids_not_found_converted.shape[0], predictions_converted.shape[1]))), axis=1)
        # Combining predictions of users found and not found
        user_predictions_converted = np.vstack([user_predictions_found_converted, user_predictions_not_found_converted])        
        user_predictions_df = pd.DataFrame(user_predictions_converted, columns=['user_id', *[str(i) for i in range(predictions_converted.shape[1])]]).set_index('user_id')
        # Ensures predictions output dataframe is sorted the same as input user_ids order
        user_predictions_df = user_predictions_df.loc[user_ids['user_id'].values]
        self.user_predictions_df = user_predictions_df
        print("Finish prediction")
        return user_predictions_df
    
    def convert_user_ids(self, user_ids):
        """Converts the encoded user ids into the original ids"""
        raw_user_ids = self.user_ids_mapping_df['user_id'].loc[user_ids].values
        return raw_user_ids
    
    def convert_prediction_item_ids(self,predictions):
        """Converts the encoded predicted item ids into the original item ids"""
        raw_topk_predicted_item_ids = self.track_ids_mapping_df['track_id'].loc[predictions.reshape(-1)].values
        raw_topk_predicted_item_ids = np.reshape(raw_topk_predicted_item_ids, (-1, predictions.shape[1]))
        return raw_topk_predicted_item_ids


# Matrix Factorization

class MyMFModel(MyRetrievalModel):
    
    def get_model(self):      
        item_retrieval_task = self.get_item_retrieval_task()

        model = mm.MatrixFactorizationModel(
            self.schema,
            dim=self.hparams['mf_dim'],
            prediction_tasks=item_retrieval_task,
            embeddings_l2_reg=self.hparams['embeddings_l2_reg']
        )
        
        return model

mf_model = MyMFModel(
    items_df=dataset.df_tracks,
    users_df=dataset.df_users,
    
    # Training hparams
    epochs=5,
    train_batch_size=8192,
    lr=1e-3,
    lr_decay_steps=100,
    lr_decay_rate=0.96,
    label_smoothing=0.0,
    
    # Model hparams
    logq_correction_factor=1.0,
    embeddings_l2_reg=5e-6,
    logits_temperature=1.8,
    mf_dim=128,
)

runner.evaluate(model=mf_model, limit=LIMIT)

END_TIME = time.time()
print(f"Matrix factorization took {(END_TIME-START_TIME) / 60} minutes!")

START_TIME = time.time()

# Two-tower architecture

class MyTwoTowerModel(MyRetrievalModel):
    
    def get_model(self):   
        item_retrieval_task = self.get_item_retrieval_task()
        
        model = mm.TwoTowerModel(
            self.schema,
            query_tower=mm.MLPBlock(
                self.hparams['tt_mlp_layers'],
                activation=self.hparams['tt_mlp_activation'],
                no_activation_last_layer=True,    
                dropout=self.hparams['tt_mlp_dropout'],                
                kernel_regularizer=regularizers.l2(self.hparams['tt_mlp_l2_reg']),
                bias_regularizer=regularizers.l2(self.hparams['tt_mlp_l2_reg']),
            ),
            embedding_options=mm.EmbeddingOptions(
                infer_embedding_sizes=True,
                infer_embedding_sizes_multiplier=self.hparams['tt_infer_embedding_sizes_multiplier'],
                embeddings_l2_reg=self.hparams['embeddings_l2_reg'],
            ),
            prediction_tasks=item_retrieval_task
        )

        return model

tt_model = MyTwoTowerModel(
    items_df=dataset.df_tracks,
    users_df=dataset.df_users,
    
    # Training hparams
    epochs=5,
    train_batch_size=8192,
    lr=1e-3,
    lr_decay_steps=100,
    lr_decay_rate=0.96,
    label_smoothing=0.0,
    
    # Model hparams
    logq_correction_factor=1.0,
    embeddings_l2_reg=1e-5,
    logits_temperature=1.8,
    tt_mlp_layers=[128,64],
    tt_mlp_activation="relu",
    tt_mlp_dropout=0.3,
    tt_mlp_l2_reg=5e-5,
    tt_infer_embedding_sizes_multiplier=2.0
)


runner.evaluate(model=tt_model, limit=LIMIT)

END_TIME = time.time()
print(f"Two tower took {(END_TIME-START_TIME) / 60} minutes!")

