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

<template><div>
  <h3 id="waiting-msg">Waiting for server...</h3>
  <div id="table">
    <h2 style="text-align:center;">Pitch accuracy (%)</h2>
    <div id="legend">
      <div class="legend-item">
        <div class="score"></div>
        <div>Train set</div>
      </div>
      <div class="legend-item">
        <div class="score validation"></div>
        <div>Live set</div>
      </div>
    </div>
    <div id="table-rows"></div>
    <button id="live-button">Test Live</button>
  </div>
</div></template>

<script lang="ts" src="./app.ts"></script>

<style>
#table {
  width: 660px;
  display: none;
}
#table-rows {
  border-right: 2px solid #bbb;
}
#table .row {
  display: flex;
  align-items: center;
  margin: 25px 0;
}
#legend {
  position: absolute;
}
.legend-item {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.legend-item .score {
  width: 30px;
  margin-right: 10px;
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
  width: 360px;
  text-align: right;
}
#table .score {
  background-color: #0277bd;
  height: 30px;
  text-align: right;
  line-height: 30px;
  color: white;
  padding-right: 10px;
  box-sizing: border-box;
}
#table .score.validation {
  background-color: #ef6c00;
}

html,
body {
  font-family: Roboto, sans-serif;
  color: #5f6368;
}

body {
  background-color: rgb(248, 249, 250);
}

#accuracyCanvas > div {
  display: none;
}

#live-button {
  padding: 10px;
  font-size: 24px;
}
</style>
