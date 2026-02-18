import pandas as pd 
import numpy as np
import os, sys
import ast

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

bc5cdr_root = os.path.join(project_root, 'data', 'external', 'bc5cdr', 'data', 'training')

class BC5CDR:
    def __init__(self):
        pass

    def parse_entity(self, file_type = 'Training'):
        '''
        Parse annotated information dataset from text file into Df-formatted 

        Input: 
            file_type: str (file name, either Training/Test/Development)
        Output: 
            df: contains 4 columns
                number: ID of the paper (containing the abstract and title)
                title: Title of the paper
                abstract: abstract of the paper
                entities: dictionary of annotated entities inside the abstract
                    text: entity name
                    type: entity type
                    start: start index (character, not word)
                    end: end character
                    mesh: ID of the entity
        '''
        file_name = f'CDR_{file_type}Set.PubTator.txt'

        data_path = os.path.join(bc5cdr_root, file_name)

        documents = {}
        current_pmid = None

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # TITLE / ABSTRACT
                if "|t|" in line or "|a|" in line:
                    pmid, rest = line.split("|", 1)
                    tag, text = rest.split("|", 1)

                    if pmid not in documents:
                        documents[pmid] = {
                            "title": "",
                            "abstract": "",
                            "entities": []
                        }

                    if tag == "t":
                        documents[pmid]["title"] = text
                    elif tag == "a":
                        documents[pmid]["abstract"] = text

                # ENTITY LINE
                else:
                    parts = line.split("\t")

                    if len(parts) == 6:
                        pmid, start, end, text, ent_type, mesh = parts

                        entity = {
                            "text": text,
                            "type": ent_type,
                            "start": int(start),
                            "end": int(end),
                            "mesh": mesh
                        }

                        documents[pmid]["entities"].append(entity)

        rows = [
            {
                "number": k,
                "title": v["title"],
                "abstract": v["abstract"],
                "entities": v["entities"]
            }
            for k, v in documents.items()
        ]

        return pd.DataFrame(rows)

    def create_lookup_table(self, df):
        '''
        Create entity-type-mesh lookup table from the annotation for further usage

        Input: 
            df: DataFrame with columns (entities: dict), actually the output of the function above

        Output:
            lookup_table: DataFrame with lookup table of annotated entities
                text: name of the entity
                type: type of the entity
                mesh: ID of the entity
        '''

        df['entities'] = df['entities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        exploded_df = df.explode('entities')

        exploded_df = exploded_df.dropna(subset=['entities'])

        entities_df = exploded_df['entities'].apply(pd.Series)
        
        # create the lookup table
        lookup_table = entities_df[['text', 'type', 'mesh']].copy()
        
        lookup_table.columns = ['Text', 'Type', 'Mesh']

        lookup_table['Text'] = lookup_table['Text'].apply(lambda x: str(x).capitalize())
        
        lookup_table = lookup_table.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Deal with combined Mesh
        df_long = (
            lookup_table
            .loc[lookup_table["Mesh"].str.len() == 15]
            .assign(Mesh=lambda x: x["Mesh"].str.split("|"))
            .explode("Mesh", ignore_index=True)
        )

        lookup_table = lookup_table[lookup_table['Mesh'].str.len() < 15]

        lookup_table = pd.concat([lookup_table, df_long], axis = 0)

        # Deal with unknown Mesh
        lookup_table['Mesh'] = lookup_table['Mesh'].apply(lambda x: 'Unknown' if x == '-1' else x)

        lookup_table = lookup_table.drop_duplicates().reset_index(drop=True)
        
        return lookup_table

    def extract_relations(self, file_type = 'Training'):
        '''
        Extract Relations from the bc5cdr annotated files

        Input: 
            file_type: str (file name, either Training/Test/Development)
        Output: 
            df: contains 3 columns
                number: ID of the paper (containing the abstract and title)
                chemical: root chemical that causes disease
                disease: diseased influenced by the chemical 
        '''
        rows = []

        file_name = f'CDR_{file_type}Set.PubTator.txt'

        data_path = os.path.join(bc5cdr_root, file_name)

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if "\tCID\t" in line:
                    pmid, _, chem, disease = line.strip().split("\t")
                    rows.append({
                        "number": pmid,
                        "chemical": chem,
                        "disease": disease
                    })

        return pd.DataFrame(rows, columns=["number", "chemical", "disease"])

