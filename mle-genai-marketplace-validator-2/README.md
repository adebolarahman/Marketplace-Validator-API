# GenAI CoE • Machine Learning Engineer - GenAI • Interview Challenge 

> This challenge is not necessarily realistic. It is designed to assess your ML Engineering & GenAI skills (an LLM may not be the best solution for this problem in reality).

## Overview

Consider a marketplace that you are developing for consumers to compare credit cards, loans, etc. 

As part of setting up the marketplace third-party lenders provide offers to list to consumers. Each lender can create their listing and description as they wish, for example,

> **Card Name:** ClearPath Platinum by TrustBank
> 
> **Brief Description:**
> Simplify your finances with the ClearPath Platinum credit card from TrustBank. Earn generous rewards on everyday purchases, including 3% cashback on dining, 2% cashback on gas and groceries, and 1% cashback everywhere else. With no annual fees, clear terms, and straightforward rewards, managing your credit has never been easier.
> 
> **Features:**
> 
> - **Cashback:** 3% on dining, 2% on gas and groceries, 1% on other purchases
> 
> - **Annual Fee:** £0
> 
> - **APR:** 17.49%–25.49% variable APR
> 
> - **Introductory Offer:** 0% APR on purchases and balance transfers for the first 15 months
> 
> - **Credit Requirement:** Good to Excellent (680+ credit score)
> 
> **Ideal For:**
> Consumers who prefer transparent rewards, straightforward terms, and consistent savings on everyday spending.

which will then appear on the marketplace. 

As the marketplace administrator, you want to have control over the way that credit cards are listed to ensure the  [Marketplace Standards](./marketplace-standards.md) are met for all listings from all lenders. 

You are tasked with building a _Marketplace Validator_ that will validate a listing from a lender and suggest changes if required.

## Tasks

Select **one** of the challenges below:

1. **Marketplace Validator API** 

   Setup a production ready API for serving the provided Marketplace Validator. 

   The API you create will be called by the lenders to validate their listing is compliant with the marketplace.
2. **Improved Marketplace Validator** 

   Improve the existing Marketplace Validator so that it performs better. Experiment with any approach you see fit. 

<br>

> Complete as much as you can in ~4 hours, there is not a requirement to complete it.

<br>

### 1. Marketplace Validator API

Build a production ready API that validates listings meet the [Marketplace Standards](./marketplace-standards.md). Where the listing does not meet standards, provide suggested changes.

You can use any LLM provider that you would like, if you would like a free option, you can set up a Gemini API Key with the AI studio https://aistudio.google.com/. Once you have that you should be able to make calls to Gemini models.

For the purposes of this challenge, you can use the [prompt provided](marketplace_validator_prompt.txt).

Your deliverables: 
- [ ] Working API running locally.
- [ ] Document any assumptions you have made.
- [ ] High level design for how to serve this in a production cloud environment.

### 2. Improved Marketplace Validator

Take the existing Marketplace Validator (just the [prompt provided](marketplace_validator_prompt.txt)) and find possible improvements.

Consider how this improved validator could be served in production.

Your deliverables:
- [ ] Notebook / scripts / any code produced. 
- [ ] Document any considerations for choosing whether or not to use the improved validator