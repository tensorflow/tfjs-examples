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
    <div class="row">
      <button type="button" class="btn btn-primary" v-on:click="loadLive">
        Load Live Data
      </button>
      <div id="accuracyCanvas"></div>
    </div>

    <div class="row d-flex">
      <div class="col-lg-4">
        <div class="card" v-if="predictions.length === 0">
          <div class="card-body text-center" style="color: #80868b">
            <span v-if="connected">Waiting for live pitch data...</span>
            <span v-if="!connected">Waiting for pitch server...</span>
          </div>
        </div>
      </div>
    </div>

    <transition-group name="flip-list">
      <pitch class="pitch"
            v-for="prediction in predictions"
            :key="prediction.uuid"
            v-bind:prediction="prediction"></pitch>
    </transition-group>
  </div>
</template>

<script lang="ts" src="./app.ts"></script>

<style>
.html {
  font-family: Roboto, sans-serif;
}
.flip-list-move {
  transition: transform 1s;
}

body {
  background-color: rgb(248, 249, 250);
}

.flip-list-move {
  transition: transform 1s;
}

.list-complete-item {
  transition: all 1s;
  display: inline-block;
  margin-right: 10px;
}
.list-complete-enter,
.list-complete-leave-to {
  opacity: 0;
  transform: translateY(20px);
}
.list-complete-leave-active {
  position: absolute;
}

.tfjs-navbar {
  background-color: #ffffff;
  font-family: "Google Sans", sans-serif;
  font-size: 32px;
  color: #80868b;
  line-height: 24px;
  font-weight: 500;
  padding-bottom: 8px;
  text-align: center;
}

.tfjs-name {
  padding-top: 40px;
  font-size: 14px;
  line-height: 16px;
  margin-bottom: 8px;
}

.tfjs-title {
  font-size: 32px;
  line-height: 24px;
  margin-bottom: 16px;
  font-weight: 600;
}

.tfjs-subtitle {
  font-size: 14px;
  text-align: center;
  line-height: 20px;
  margin-left: auto;
  margin-right: auto;
}

.content {
  padding-top: 20px;
}

#accuracyCanvas > div {
  display: none;
}

.footer {
  position: absolute;
  bottom: 0;
  width: 100%;
  height: 60px;
  line-height: 60px;
  background-color: #f5f5f5;
}
</style>
