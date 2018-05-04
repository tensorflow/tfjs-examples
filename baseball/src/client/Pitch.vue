<template>

<div class="row d-flex">
  <div class="col-lg-4">
    <div class="card" v-bind:class="[correct ? 'correct' : 'incorrect']">
      <div class="pitch-card d-flex flex-row align-items-start">

        <div class="copy">
          <div class="label d-flex justify-content-between">
            <span>{{ prediction.pitch_classes[0].type }}</span>
            <i v-if="correct" class="material-icons correct icon">done</i>
            <i v-if="!correct" class="material-icons incorrect icon">clear</i>
          </div>
        </div>

      </div>
    </div>
  </div>
</div>

</template>

<script lang="ts" src="./pitch.ts"></script>

<style>

.card {
  transition: 1.25s;
  /* display: inline-block; */
  background: #ffffff;
  box-shadow: 0 1px 2px 0 rgba(60, 64, 67, 0.3),
    0 1px 3px 1px rgba(60, 64, 67, 0.15);
  border-radius: 8px;
  margin-bottom: 16px;
}

.card.correct {
  background-color: rgba(0, 255, 0, 0.10);
}

.card.incorrect {
  background-color: rgba(255, 0, 0, 0.1);
}

.icon.correct {
  color: #34A853;
}

.icon.incorrect {
  color: #D0021B;
}

.pitch-card {
  padding: 32px;
}

.copy {
  display: inline-block;
  width: 100%;
}

.label {
  font-family: "Google Sans", sans-serif;
  font-size: 24px;
  color: #5f6368;
  line-height: 24px;
  font-weight: 500;
}
</style>
