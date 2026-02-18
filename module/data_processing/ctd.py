import pandas as pd 
import numpy as np
import ast

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

data_dir = os.path.join(project_root, "data", 'external', 'ChemDisGene', 'data', "ctd_derived")

class CTD:
    def __init__(self):
        pass

    def process_ctd(self, file_type='train'):
        '''
        Input:
            file_type: Either 'train', 'test', or 'dev'
        Output:
            pd.DataFrame with columns: docid, title, abstract, mentions, mentions
        '''
        from .pubtator import parse_pubtator

        abstracts_path = os.path.join(data_dir, f"{file_type}_abstracts.txt")
        relations_path = os.path.join(data_dir, f"{file_type}_mentions.tsv")
        
        print(f"Reading from: {abstracts_path}")
        print(f"Reading from: {relations_path}")
        
        def clean_id(eid):
            if not eid: return None
            return eid.replace("id:", "").replace("OMIM:", "")

        # Parse the raw files
        docs = parse_pubtator(
            pbtr_file=abstracts_path,
            relns_file=relations_path
        )
        
        data = []
        
        for doc in docs:
            # Process Mentions -> List of Dicts
            doc_mentions = []
            for m in doc.mentions:
                doc_mentions.append({
                    "text": m.mention,
                    "type": m.entity_type,
                    "id": clean_id(m.entity_ids[0]) if m.entity_ids else None,
                    "start": m.ch_start,
                    "end": m.ch_end
                })
                
            # Process mentions -> List of Dicts
            doc_relations = []
            for r in doc.mentions:
                doc_relations.append({
                    "type": r.relation_label,
                    "subject_id": clean_id(r.subj_eid),
                    "object_id": clean_id(r.obj_eid),
                    "subject_type": r.subj_type,
                    "object_type": r.obj_type
                })
                
            data.append({
                "docid": doc.docid,
                "title": doc.title,
                "abstract": doc.abstract,
                "mentions": doc_mentions,
                "mentions": doc_relations
            })
            
        return pd.DataFrame(data)

    def create_lookup_table(self, df):
        '''
        Create entity-type-id lookup table from the annotation for further usage

        Input: 
            df: DataFrame with columns (mentions: dict), actually the output of the function above

        Output:
            lookup_table: DataFrame with lookup table of annotated mentions
                text: name of the entity
                type: type of the entity
                id: ID of the entity
        '''

        df['mentions'] = df['mentions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        exploded_df = df.explode('mentions')

        exploded_df = exploded_df.dropna(subset=['mentions'])

        mentions_df = exploded_df['mentions'].apply(pd.Series)
        
        # create the lookup table
        lookup_table = mentions_df[['text', 'type', 'id']].copy()
        
        lookup_table.columns = ['text', 'type', 'id']

        lookup_table['text'] = lookup_table['text'].apply(lambda x: str(x).capitalize())
        
        lookup_table = lookup_table.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Deal with combined id
        df_long = (
            lookup_table
            .loc[lookup_table["id"].str.len() == 15]
            .assign(id=lambda x: x["id"].str.split("|"))
            .explode("id", ignore_index=True)
        )

        lookup_table = lookup_table[lookup_table['id'].str.len() < 15]

        lookup_table = pd.concat([lookup_table, df_long], axis = 0)

        # Deal with unknown id
        lookup_table['id'] = lookup_table['id'].apply(lambda x: 'Unknown' if x == '-1' else x)

        lookup_table = lookup_table.drop_duplicates().reset_index(drop=True)
        
        return lookup_table

