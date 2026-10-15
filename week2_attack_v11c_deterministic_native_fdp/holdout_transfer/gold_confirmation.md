# v11c gold-answer confirmation (offline)

This joins the 39 lexical candidates to MP-DocVQA validation answers. No model or evaluator was run.

- candidate rows: 39
- clean matches gold: 35/39
- attack matches gold: 0/39
- noise matches gold: 35/39
- confirmed content failures (clean yes, attack no, noise yes): 35/39
- Full replay true for all candidate rows: True
- attack/noise outputs valid for all candidate rows: True

A confirmed row still means the model answer is changed relative to the dataset answer; it does not by itself establish a causal method advantage beyond the matched noise control.

| sample | ratio | gold | clean | attack | noise | confirmed | Full replay |
|---:|---:|---|---|---|---|:---:|:---:|
| 53993 | 0.25 | ohio; Ohio | Ohio | SUPERRAMERICA,LLC | Ohio | True | True |
| 53993 | 0.35 | ohio; Ohio | Ohio | SUPERRAMERICAC,LLC | Ohio | True | True |
| 57531 | 0.15 | Sunfeast | Sunfeast | Delicious | Sunfeast | True | True |
| 57531 | 0.25 | Sunfeast | Sunfeast | UTC | Sunfeast | True | True |
| 57531 | 0.35 | Sunfeast | Sunfeast | Wheat Rusk | Sunfeast | True | True |
| 51660 | 0.15 | May 4, 1960 | May 4, 1960 | 16/02/2007 | May 4, 1960 | True | True |
| 51660 | 0.25 | May 4, 1960 | May 4, 1960 | 16-4-1960 | May 4, 1960 | True | True |
| 51660 | 0.35 | May 4, 1960 | May 4, 1960 | 1955 | May 4, 1960 | True | True |
| 50906 | 0.15 | Dr Darby | Dr. Darby | Dr. Dir Donkey | Dr. Darby | True | True |
| 51140 | 0.25 | June 7, 2016 | June 7, 2016 | H.K.T., N.K. | June 7, 2016 | True | True |
| 51140 | 0.35 | June 7, 2016 | June 7, 2016 | H.R. 347, 114th Congress, | June 7, 2016 | True | True |
| 51140 | 0.50 | June 7, 2016 | June 7, 2016 | June 7, 2015 | June 7, 2016 | True | True |
| 15282 | 0.25 | LSE | LuxSE | BSE | LuxSE | True | True |
| 15282 | 0.35 | LSE | LuxSE | BSE | LuxSE | True | True |
| 15282 | 0.50 | LSE | LuxSE | Euronext | LuxSE | True | True |
| 790 | 0.25 | Michael F. Jacobson, Ph.D.; Michael F. Jacobson | Michael F. Jacobson, Ph.D. | J. Alexander Mc Donough | Michael F. Jacobson, Ph.D. | True | True |
| 790 | 0.35 | Michael F. Jacobson, Ph.D.; Michael F. Jacobson | Michael F. Jacobson, Ph.D. | J. Alexander M. Douglas, Jr. | Michael F. Jacobson, Ph.D. | True | True |
| 49432 | 0.25 | laboratory research division; LABORATORY RESEARCH DIVISION | Laboratory research division | Research Service Department | Laboratory research division | True | True |
| 49432 | 0.35 | laboratory research division; LABORATORY RESEARCH DIVISION | Laboratory research division | THE SAMUEL ROBBETT INC | Laboratory research division | True | True |
| 58827 | 0.25 | OFFICERS OF THE INSTITUTE; Officers of the institute | OFFICERS OF THE INSTITUTE | Chairs, Comship, and Research | OFFICERS OF THE INSTITUTE | True | True |
| 58827 | 0.35 | OFFICERS OF THE INSTITUTE; Officers of the institute | OFFICERS OF THE INSTITUTE | MEMBER SHIPS | OFFICERS OF THE INSTITUTE | True | True |
| 58421 | 0.15 | heating in sealed tube; Heating in sealed tube | HEATING IN OPEN AIR | Figure 3 | HEATING IN OPEN AIR | False | True |
| 58421 | 0.25 | heating in sealed tube; Heating in sealed tube | HEATING IN OPEN AIR | THERMOMETER | Heating in open air | False | True |
| 58421 | 0.35 | heating in sealed tube; Heating in sealed tube | Heating in open air | Thiamine | Heating in open air | False | True |
| 58421 | 0.50 | heating in sealed tube; Heating in sealed tube | Heating in open air | Heating in nitrogen sulfoxide | Heating in open air | False | True |
| 57887 | 0.25 | 1mg | 1mg | 9013 | 1mg | True | True |
| 57887 | 0.35 | 1mg | 1 mg | 8 | 1 mg | True | True |
| 62321 | 0.15 | the new york state department of social services; New York State Department of Social Services | THE NEW YORK STATE DEPARTMENT OF SOCIAL SERVICES | THE NEW YORK STATE BOARD OF SOCIAL WELFARE | THE NEW YORK STATE DEPARTMENT OF SOCIAL SERVICES | True | True |
| 62321 | 0.25 | the new york state department of social services; New York State Department of Social Services | THE NEW YORK STATE DEPARTMENT OF SOCIAL SERVICES | THE NEW YORK STATE DEPARTMENT OF SOCIAL SECURITY | THE NEW YORK STATE DEPARTMENT OF SOCIAL SERVICES | True | True |
| 25716 | 0.25 | 12:30 p.m. | 12:30 p.m. | 1 hour | 12:30 p.m. | True | True |
| 25716 | 0.35 | 12:30 p.m. | 12:30 p.m. | Under-separation | 12:30 p.m. | True | True |
| 55463 | 0.25 | cigarette smoke contains carbon monoxide.; Cigarette smoke contains Carbon monoxide | Cigarette Smoke Contains Carbon Monoxide | Bored. Lonely. | Cigarette Smoke Contains Carbon Monoxide | True | True |
| 55463 | 0.35 | cigarette smoke contains carbon monoxide.; Cigarette smoke contains Carbon monoxide | Cigarette Smoke Contains Carbon Monoxide | Smoketooth character | Cigarette Smoke Contains Carbon Monoxide | True | True |
| 64689 | 0.25 | 500 MG; 500 mg | 500 mg | N O INTAKE N U G | 500 mg | True | True |
| 64689 | 0.35 | 500 MG; 500 mg | 500 mg | O.552 | 500 mg | True | True |
| 64689 | 0.50 | 500 MG; 500 mg | 500 mg | 512.5 mg | 500 mg | True | True |
| 50312 | 0.15 | 604 | 604 | 674 | 604 | True | True |
| 50312 | 0.25 | 604 | 604 | 1 | 604 | True | True |
| 50312 | 0.35 | 604 | 604 | 1 | 604 | True | True |
