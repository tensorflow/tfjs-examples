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
  <div class="card" v-bind:class="[correct ? 'correct' : 'incorrect']">
    <div class="pitch-card">
      <div class="copy">
        <div class="label">
          <span>{{ prediction.pitch_classes[0].type }}</span>
          <i v-if="correct" class="material-icons correct icon">done</i>
          <i v-if="!correct" class="material-icons incorrect icon">clear</i>
        </div>
      </div>

    </div>
  </div>
</template>

<script lang="ts" src="./pitch.ts"></script>

<style>
.card {
  width: 400px;
  transition: 1.25s;
  background: #ffffff;
  box-shadow: 0 1px 2px 0 rgba(60, 64, 67, 0.3),
    0 1px 3px 1px rgba(60, 64, 67, 0.15);
  border-radius: 8px;
  margin-bottom: 16px;
}

.card.correct {
  background-color: rgba(0, 255, 0, 0.1);
}

.card.incorrect {
  background-color: rgba(255, 0, 0, 0.1);
}

.icon.correct {
  color: #34a853;
}

.icon.incorrect {
  color: #d0021b;
}

.pitch-card {
  padding: 32px;
}

.copy {
  display: inline-block;
  width: 100%;
}
</style>
