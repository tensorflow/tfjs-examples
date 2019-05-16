# Text Classifer Example

This example uses the universal sentence encoder to train two text
classification models.

1. An 'intent' classifier that classifies sentences into categories representing
user intent for a query.
2. A token tagger, that classifies tokens within a weather releated query to
identify location related tokens.

## Setup and Installation

Note: These instructions use `yarn`, but you can use `npm run` instead if you
do not have `yarn` installed.

Install dependencies

```
yarn
```

## Preparing training data

There are four npm/yarn scripts listed in package.json for preparing the training data. Each writes out one of more new files.

The two scripts needed to train the intent classifier are:

1. `yarn convert-raw-to-csv`: Converts the raw data into a csv format
2. `yarn convert-csv-to-tensors`: Converts the strings in the CSV created in step 1 into tensors.

The two scripts needed to train the token tagger are:

1. `yarn convert-raw-to-tagged-tokens`: Extracts tokens from sentences in the original data and tags each token with a category
2. `yarn convert-tokens-to-embeddings`: embeds the tokens from the queries using the universal sentence encoder and writes out a look-up-table.

You can run all four of these commands with

```
yarn prep-data
```

You only need to do this once. This process can take 15-25 mins. The output of these scripts will be written to the `training/data` folder.

## Train the models

To train the intent classifier model run:

```
yarn train-intent
```

To train the token tagging model run:

```
yarn train-tagger
```

Each of these scripts take multiple options, look at `training/train-intent.js` and `training/train-tagger.js` for details.

These scripts will output model artifacts in the `training/models` folder.

## Run the apps

Once the models are trained you can use the following commands
to run the demo apps for each model.


```
yarn intent-app
```


```
yarn tagger-app
```
