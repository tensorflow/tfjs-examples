import Vue from 'vue';

// tslint:disable-next-line:no-default-export
export default Vue.component('pitch', {
  props: ['prediction'],

  computed: {
    correct: function() {
      return this.prediction.pitch_classes[0].pitch_code ===
          this.prediction.pitch.pitch_code;
    }
  }
});
