<!--
==============================================================================
Copyright 2018 Google LLC. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
-->

<template>
  <div class="container content">
    <div id="table"></div>
  </div>
</template>

<script lang="ts" src="./app.ts"></script>

<style>
#table .row {
  display: flex;
  align-items: center;
  margin: 5px 0;
}
.label {
  text-align: center;
  font-family: "Google Sans", sans-serif;
  font-size: 24px;
  color: #5f6368;
  line-height: 24px;
  font-weight: 500;
}
#table .label {
  margin-right: 20px;
  width: 300px;
  text-align: right;
}
#table .score {
  background-color: #999;
  height: 30px;
  text-align: right;
  line-height: 30px;
  color: white;
  padding-right: 10px;
  box-sizing: border-box;
}

#table .score-container {
  border-right: 1px solid black;
}

html,
body {
  font-family: Roboto, sans-serif;
}

body {
  background-color: rgb(248, 249, 250);
}

#accuracyCanvas > div {
  display: none;
}
</style>
