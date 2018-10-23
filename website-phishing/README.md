# TensorFlow.js Example: Classify Website URLs as Phishy or Normal

This example shows you how to classify URLs as [phishy](https://en.wikipedia.org/wiki/Phishing) or normal using [Phishing Website Dataset](http://eprints.hud.ac.uk/id/eprint/24330/6/MohammadPhishing14July2015.pdf). Since we are classifying the elements of a given set into two groups ie. phishy or normal, this is a binary classification problem.

## Features

We have following features available in the dataset:

1. `HAVING_IP_ADDRESS`: Whether an IP adress is used as an alternate to a domain name. {-1, 1}
2. `URL_LENGTH`: Whether URL length is legitimate, suspicious or phishing. {1, 0, -1}
3. `SHORTINING_SERVICE`: Whether it is using URL shortening service or not. {1, -1}
4. `HAVING_AT_SYMBOL`: Whether URL contains "@" symbol. {-1, 1}
5. `DOUBLE_SLASH_REDIRECTING`: Whether URL contains double slash redirecting or not. {-1, 1}
6. `PREFIX_SUFFIX`: Whether URL contains prefix or suffix separated by "-". {-1, 1}
7. `HAVING_SUB_DOMAIN`: Whether count of sub domains in URL is legitimate, suspicious or phishing. {-1, 0, 1}
8. `SSLFINAL_STATE`: Whether URL use https and issuer is trusted, use https but issuer is not trusted or no https. {-1, 0, 1}
9. `DOMAIN_REGISTERATION_LENGTH`: Whether domain expires in less than a year or not. {-1, 1}
10. `FAVICON`: Whether favicon is loaded from external domain or not. {-1, 1}
11. `PORT`: Whether port is of preferred status or not. {-1, 1}
12. `HTTPS_TOKEN`: Whether `https` token is part of domain or not. {-1, 1}
13. `REQUEST_URL`: Whether percentage of requests made to external domain falls in legitimate or suspicious category. {-1, 1}
14. `URL_OF_ANCHOR`: Whether percentage of url in anchor tags reference external domain or self falls in legitimate, suspicious or phishy category. {-1, 0, 1}
15. `LINKS_IN_TAGS`: Whether percentage of links in meta, script, link tags referencing external domain falls in legitimate, suspicious or phishy category. {-1, 0, 1}
16. `SFH`: Whether server form handler is empty or contains "about: blank", refers to a different domain or is normal. {1, 0, -1}
17. `SUBMITTING_TO_EMAIL`: Whether the form submits information to email. {-1, 1}
18. `ABNORMAL_URL`: Whether URL contains host name or not. {-1, 1}
19. `REDIRECT`: Whether URL redirects less than equal to 1, between 2 and 4 or greater than 4. {0, 1}
20. `ON_MOUSEOVER`: Whether `onMouseOver` changes status bar or not. {-1, 1}
21. `RIGHTCLICK`: Whether right click is disabled or not. {-1, 1}
22. `POPUPWIDNOW`: Whether pop up window contain text field or not. {-1, 1}
23. `IFRAME`: Whether page contains iframe tag or not. {-1, 1}
24. `AGE_OF_DOMAIN`: Whether age of domain is less than 6 months or not. {-1, 1}
25. `DNSRECORD`: Whether there is DNS record for the domain or not. {-1, 1}
26. `WEB_TRAFFIC`: Whether website ranking is less than 100,000, greater than 100,000 or is not recognized by Alexa and/or has no web traffic. {-1, 0, 1}
27. `PAGE_RANK`: Whether page rank is less than 0.2 or not. {-1, 1}
28. `GOOGLE_INDEX`: Whether web page is indexed by Google or not. {-1, 1}
29. `LINKS_POINTING_TO_PAGE`: Whether links pointing to the page is equal to 0, between 0 and 2 or greater than 2. {-1, 0, 1}
30. `STATISTICAL_REPORT`: Host belongs to top phishing IPs or domains or not. {-1, 1}

## Usage

Prepare the environment:
```sh
$ npm install
# Or
$ yarn
```

To build and watch the example, run:
```sh
$ yarn watch
```
