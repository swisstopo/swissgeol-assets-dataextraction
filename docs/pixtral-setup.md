# AWS Setup for Pixtral Classifier

To run classification using the Pixtral Large model, you must configure your AWS credentials for Amazon Bedrock.

## Prerequisites

- Access to Amazon Bedrock and the Pixtral model
- AWS credentials configured

## Credential Configuration

### Option A: AWS CLI

```bash
aws configure
```

### Option B: Manual Config Files

Create or edit the following files:

**~/.aws/config**

```
[default]
region=eu-central-1
output=json
```

**~/.aws/credentials**

```
[default]
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
```

## Running Classification with Pixtral

```bash
python main.py -i <input_path> -g <ground_truth_path> -c pixtral
```

## Configuration

The Pixtral classifier is configured via `config/pixtral_config.yml`.
