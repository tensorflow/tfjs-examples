/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core'], factory) :
    (factory((global.tf = global.tf || {}, global.tf.data = global.tf.data || {}),global.tf));
}(this, (function (exports,tf) { 'use strict';

    /*! *****************************************************************************
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the Apache License, Version 2.0 (the "License"); you may not use
    this file except in compliance with the License. You may obtain a copy of the
    License at http://www.apache.org/licenses/LICENSE-2.0

    THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
    WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
    MERCHANTABLITY OR NON-INFRINGEMENT.

    See the Apache Version 2.0 License for specific language governing permissions
    and limitations under the License.
    ***************************************************************************** */
    /* global Reflect, Promise */

    var extendStatics = function(d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };

    function __extends(d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    }

    function __awaiter(thisArg, _arguments, P, generator) {
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    }

    function __generator(thisArg, body) {
        var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
        return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
        function verb(n) { return function (v) { return step([n, v]); }; }
        function step(op) {
            if (f) throw new TypeError("Generator is already executing.");
            while (_) try {
                if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
                if (y = 0, t) op = [op[0] & 2, t.value];
                switch (op[0]) {
                    case 0: case 1: t = op; break;
                    case 4: _.label++; return { value: op[1], done: false };
                    case 5: _.label++; y = op[1]; op = [0]; continue;
                    case 7: op = _.ops.pop(); _.trys.pop(); continue;
                    default:
                        if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                        if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                        if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                        if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                        if (t[2]) _.ops.pop();
                        _.trys.pop(); continue;
                }
                op = body.call(thisArg, _);
            } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
            if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
        }
    }

    var commonjsGlobal = typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};

    function unwrapExports (x) {
    	return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, 'default') ? x['default'] : x;
    }

    function createCommonjsModule(fn, module) {
    	return module = { exports: {} }, fn(module, module.exports), module.exports;
    }

    var alea = createCommonjsModule(function (module) {
    // A port of an algorithm by Johannes Baagøe <baagoe@baagoe.com>, 2010
    // http://baagoe.com/en/RandomMusings/javascript/
    // https://github.com/nquinlan/better-random-numbers-for-javascript-mirror
    // Original work is under MIT license -

    // Copyright (C) 2010 by Johannes Baagøe <baagoe@baagoe.org>
    //
    // Permission is hereby granted, free of charge, to any person obtaining a copy
    // of this software and associated documentation files (the "Software"), to deal
    // in the Software without restriction, including without limitation the rights
    // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    // copies of the Software, and to permit persons to whom the Software is
    // furnished to do so, subject to the following conditions:
    // 
    // The above copyright notice and this permission notice shall be included in
    // all copies or substantial portions of the Software.
    // 
    // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    // THE SOFTWARE.



    (function(global, module, define) {

    function Alea(seed) {
      var me = this, mash = Mash();

      me.next = function() {
        var t = 2091639 * me.s0 + me.c * 2.3283064365386963e-10; // 2^-32
        me.s0 = me.s1;
        me.s1 = me.s2;
        return me.s2 = t - (me.c = t | 0);
      };

      // Apply the seeding algorithm from Baagoe.
      me.c = 1;
      me.s0 = mash(' ');
      me.s1 = mash(' ');
      me.s2 = mash(' ');
      me.s0 -= mash(seed);
      if (me.s0 < 0) { me.s0 += 1; }
      me.s1 -= mash(seed);
      if (me.s1 < 0) { me.s1 += 1; }
      me.s2 -= mash(seed);
      if (me.s2 < 0) { me.s2 += 1; }
      mash = null;
    }

    function copy(f, t) {
      t.c = f.c;
      t.s0 = f.s0;
      t.s1 = f.s1;
      t.s2 = f.s2;
      return t;
    }

    function impl(seed, opts) {
      var xg = new Alea(seed),
          state = opts && opts.state,
          prng = xg.next;
      prng.int32 = function() { return (xg.next() * 0x100000000) | 0; };
      prng.double = function() {
        return prng() + (prng() * 0x200000 | 0) * 1.1102230246251565e-16; // 2^-53
      };
      prng.quick = prng;
      if (state) {
        if (typeof(state) == 'object') copy(state, xg);
        prng.state = function() { return copy(xg, {}); };
      }
      return prng;
    }

    function Mash() {
      var n = 0xefc8249d;

      var mash = function(data) {
        data = data.toString();
        for (var i = 0; i < data.length; i++) {
          n += data.charCodeAt(i);
          var h = 0.02519603282416938 * n;
          n = h >>> 0;
          h -= n;
          h *= n;
          n = h >>> 0;
          h -= n;
          n += h * 0x100000000; // 2^32
        }
        return (n >>> 0) * 2.3283064365386963e-10; // 2^-32
      };

      return mash;
    }


    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() { return impl; });
    } else {
      this.alea = impl;
    }

    })(
      commonjsGlobal,
      module,    // present in node.js
      (typeof undefined) == 'function'   // present with an AMD loader
    );
    });

    var xor128 = createCommonjsModule(function (module) {
    // A Javascript implementaion of the "xor128" prng algorithm by
    // George Marsaglia.  See http://www.jstatsoft.org/v08/i14/paper

    (function(global, module, define) {

    function XorGen(seed) {
      var me = this, strseed = '';

      me.x = 0;
      me.y = 0;
      me.z = 0;
      me.w = 0;

      // Set up generator function.
      me.next = function() {
        var t = me.x ^ (me.x << 11);
        me.x = me.y;
        me.y = me.z;
        me.z = me.w;
        return me.w ^= (me.w >>> 19) ^ t ^ (t >>> 8);
      };

      if (seed === (seed | 0)) {
        // Integer seed.
        me.x = seed;
      } else {
        // String seed.
        strseed += seed;
      }

      // Mix in string seed, then discard an initial batch of 64 values.
      for (var k = 0; k < strseed.length + 64; k++) {
        me.x ^= strseed.charCodeAt(k) | 0;
        me.next();
      }
    }

    function copy(f, t) {
      t.x = f.x;
      t.y = f.y;
      t.z = f.z;
      t.w = f.w;
      return t;
    }

    function impl(seed, opts) {
      var xg = new XorGen(seed),
          state = opts && opts.state,
          prng = function() { return (xg.next() >>> 0) / 0x100000000; };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11,
              bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (typeof(state) == 'object') copy(state, xg);
        prng.state = function() { return copy(xg, {}); };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() { return impl; });
    } else {
      this.xor128 = impl;
    }

    })(
      commonjsGlobal,
      module,    // present in node.js
      (typeof undefined) == 'function'   // present with an AMD loader
    );
    });

    var xorwow = createCommonjsModule(function (module) {
    // A Javascript implementaion of the "xorwow" prng algorithm by
    // George Marsaglia.  See http://www.jstatsoft.org/v08/i14/paper

    (function(global, module, define) {

    function XorGen(seed) {
      var me = this, strseed = '';

      // Set up generator function.
      me.next = function() {
        var t = (me.x ^ (me.x >>> 2));
        me.x = me.y; me.y = me.z; me.z = me.w; me.w = me.v;
        return (me.d = (me.d + 362437 | 0)) +
           (me.v = (me.v ^ (me.v << 4)) ^ (t ^ (t << 1))) | 0;
      };

      me.x = 0;
      me.y = 0;
      me.z = 0;
      me.w = 0;
      me.v = 0;

      if (seed === (seed | 0)) {
        // Integer seed.
        me.x = seed;
      } else {
        // String seed.
        strseed += seed;
      }

      // Mix in string seed, then discard an initial batch of 64 values.
      for (var k = 0; k < strseed.length + 64; k++) {
        me.x ^= strseed.charCodeAt(k) | 0;
        if (k == strseed.length) {
          me.d = me.x << 10 ^ me.x >>> 4;
        }
        me.next();
      }
    }

    function copy(f, t) {
      t.x = f.x;
      t.y = f.y;
      t.z = f.z;
      t.w = f.w;
      t.v = f.v;
      t.d = f.d;
      return t;
    }

    function impl(seed, opts) {
      var xg = new XorGen(seed),
          state = opts && opts.state,
          prng = function() { return (xg.next() >>> 0) / 0x100000000; };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11,
              bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (typeof(state) == 'object') copy(state, xg);
        prng.state = function() { return copy(xg, {}); };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() { return impl; });
    } else {
      this.xorwow = impl;
    }

    })(
      commonjsGlobal,
      module,    // present in node.js
      (typeof undefined) == 'function'   // present with an AMD loader
    );
    });

    var xorshift7 = createCommonjsModule(function (module) {
    // A Javascript implementaion of the "xorshift7" algorithm by
    // François Panneton and Pierre L'ecuyer:
    // "On the Xorgshift Random Number Generators"
    // http://saluc.engr.uconn.edu/refs/crypto/rng/panneton05onthexorshift.pdf

    (function(global, module, define) {

    function XorGen(seed) {
      var me = this;

      // Set up generator function.
      me.next = function() {
        // Update xor generator.
        var X = me.x, i = me.i, t, v;
        t = X[i]; t ^= (t >>> 7); v = t ^ (t << 24);
        t = X[(i + 1) & 7]; v ^= t ^ (t >>> 10);
        t = X[(i + 3) & 7]; v ^= t ^ (t >>> 3);
        t = X[(i + 4) & 7]; v ^= t ^ (t << 7);
        t = X[(i + 7) & 7]; t = t ^ (t << 13); v ^= t ^ (t << 9);
        X[i] = v;
        me.i = (i + 1) & 7;
        return v;
      };

      function init(me, seed) {
        var j, w, X = [];

        if (seed === (seed | 0)) {
          // Seed state array using a 32-bit integer.
          w = X[0] = seed;
        } else {
          // Seed state using a string.
          seed = '' + seed;
          for (j = 0; j < seed.length; ++j) {
            X[j & 7] = (X[j & 7] << 15) ^
                (seed.charCodeAt(j) + X[(j + 1) & 7] << 13);
          }
        }
        // Enforce an array length of 8, not all zeroes.
        while (X.length < 8) X.push(0);
        for (j = 0; j < 8 && X[j] === 0; ++j);
        if (j == 8) w = X[7] = -1; else w = X[j];

        me.x = X;
        me.i = 0;

        // Discard an initial 256 values.
        for (j = 256; j > 0; --j) {
          me.next();
        }
      }

      init(me, seed);
    }

    function copy(f, t) {
      t.x = f.x.slice();
      t.i = f.i;
      return t;
    }

    function impl(seed, opts) {
      if (seed == null) seed = +(new Date);
      var xg = new XorGen(seed),
          state = opts && opts.state,
          prng = function() { return (xg.next() >>> 0) / 0x100000000; };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11,
              bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (state.x) copy(state, xg);
        prng.state = function() { return copy(xg, {}); };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() { return impl; });
    } else {
      this.xorshift7 = impl;
    }

    })(
      commonjsGlobal,
      module,    // present in node.js
      (typeof undefined) == 'function'   // present with an AMD loader
    );
    });

    var xor4096 = createCommonjsModule(function (module) {
    // A Javascript implementaion of Richard Brent's Xorgens xor4096 algorithm.
    //
    // This fast non-cryptographic random number generator is designed for
    // use in Monte-Carlo algorithms. It combines a long-period xorshift
    // generator with a Weyl generator, and it passes all common batteries
    // of stasticial tests for randomness while consuming only a few nanoseconds
    // for each prng generated.  For background on the generator, see Brent's
    // paper: "Some long-period random number generators using shifts and xors."
    // http://arxiv.org/pdf/1004.3115v1.pdf
    //
    // Usage:
    //
    // var xor4096 = require('xor4096');
    // random = xor4096(1);                        // Seed with int32 or string.
    // assert.equal(random(), 0.1520436450538547); // (0, 1) range, 53 bits.
    // assert.equal(random.int32(), 1806534897);   // signed int32, 32 bits.
    //
    // For nonzero numeric keys, this impelementation provides a sequence
    // identical to that by Brent's xorgens 3 implementaion in C.  This
    // implementation also provides for initalizing the generator with
    // string seeds, or for saving and restoring the state of the generator.
    //
    // On Chrome, this prng benchmarks about 2.1 times slower than
    // Javascript's built-in Math.random().

    (function(global, module, define) {

    function XorGen(seed) {
      var me = this;

      // Set up generator function.
      me.next = function() {
        var w = me.w,
            X = me.X, i = me.i, t, v;
        // Update Weyl generator.
        me.w = w = (w + 0x61c88647) | 0;
        // Update xor generator.
        v = X[(i + 34) & 127];
        t = X[i = ((i + 1) & 127)];
        v ^= v << 13;
        t ^= t << 17;
        v ^= v >>> 15;
        t ^= t >>> 12;
        // Update Xor generator array state.
        v = X[i] = v ^ t;
        me.i = i;
        // Result is the combination.
        return (v + (w ^ (w >>> 16))) | 0;
      };

      function init(me, seed) {
        var t, v, i, j, w, X = [], limit = 128;
        if (seed === (seed | 0)) {
          // Numeric seeds initialize v, which is used to generates X.
          v = seed;
          seed = null;
        } else {
          // String seeds are mixed into v and X one character at a time.
          seed = seed + '\0';
          v = 0;
          limit = Math.max(limit, seed.length);
        }
        // Initialize circular array and weyl value.
        for (i = 0, j = -32; j < limit; ++j) {
          // Put the unicode characters into the array, and shuffle them.
          if (seed) v ^= seed.charCodeAt((j + 32) % seed.length);
          // After 32 shuffles, take v as the starting w value.
          if (j === 0) w = v;
          v ^= v << 10;
          v ^= v >>> 15;
          v ^= v << 4;
          v ^= v >>> 13;
          if (j >= 0) {
            w = (w + 0x61c88647) | 0;     // Weyl.
            t = (X[j & 127] ^= (v + w));  // Combine xor and weyl to init array.
            i = (0 == t) ? i + 1 : 0;     // Count zeroes.
          }
        }
        // We have detected all zeroes; make the key nonzero.
        if (i >= 128) {
          X[(seed && seed.length || 0) & 127] = -1;
        }
        // Run the generator 512 times to further mix the state before using it.
        // Factoring this as a function slows the main generator, so it is just
        // unrolled here.  The weyl generator is not advanced while warming up.
        i = 127;
        for (j = 4 * 128; j > 0; --j) {
          v = X[(i + 34) & 127];
          t = X[i = ((i + 1) & 127)];
          v ^= v << 13;
          t ^= t << 17;
          v ^= v >>> 15;
          t ^= t >>> 12;
          X[i] = v ^ t;
        }
        // Storing state as object members is faster than using closure variables.
        me.w = w;
        me.X = X;
        me.i = i;
      }

      init(me, seed);
    }

    function copy(f, t) {
      t.i = f.i;
      t.w = f.w;
      t.X = f.X.slice();
      return t;
    }
    function impl(seed, opts) {
      if (seed == null) seed = +(new Date);
      var xg = new XorGen(seed),
          state = opts && opts.state,
          prng = function() { return (xg.next() >>> 0) / 0x100000000; };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11,
              bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (state.X) copy(state, xg);
        prng.state = function() { return copy(xg, {}); };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() { return impl; });
    } else {
      this.xor4096 = impl;
    }

    })(
      commonjsGlobal,                                     // window object or global
      module,    // present in node.js
      (typeof undefined) == 'function'   // present with an AMD loader
    );
    });

    var tychei = createCommonjsModule(function (module) {
    // A Javascript implementaion of the "Tyche-i" prng algorithm by
    // Samuel Neves and Filipe Araujo.
    // See https://eden.dei.uc.pt/~sneves/pubs/2011-snfa2.pdf

    (function(global, module, define) {

    function XorGen(seed) {
      var me = this, strseed = '';

      // Set up generator function.
      me.next = function() {
        var b = me.b, c = me.c, d = me.d, a = me.a;
        b = (b << 25) ^ (b >>> 7) ^ c;
        c = (c - d) | 0;
        d = (d << 24) ^ (d >>> 8) ^ a;
        a = (a - b) | 0;
        me.b = b = (b << 20) ^ (b >>> 12) ^ c;
        me.c = c = (c - d) | 0;
        me.d = (d << 16) ^ (c >>> 16) ^ a;
        return me.a = (a - b) | 0;
      };

      /* The following is non-inverted tyche, which has better internal
       * bit diffusion, but which is about 25% slower than tyche-i in JS.
      me.next = function() {
        var a = me.a, b = me.b, c = me.c, d = me.d;
        a = (me.a + me.b | 0) >>> 0;
        d = me.d ^ a; d = d << 16 ^ d >>> 16;
        c = me.c + d | 0;
        b = me.b ^ c; b = b << 12 ^ d >>> 20;
        me.a = a = a + b | 0;
        d = d ^ a; me.d = d = d << 8 ^ d >>> 24;
        me.c = c = c + d | 0;
        b = b ^ c;
        return me.b = (b << 7 ^ b >>> 25);
      }
      */

      me.a = 0;
      me.b = 0;
      me.c = 2654435769 | 0;
      me.d = 1367130551;

      if (seed === Math.floor(seed)) {
        // Integer seed.
        me.a = (seed / 0x100000000) | 0;
        me.b = seed | 0;
      } else {
        // String seed.
        strseed += seed;
      }

      // Mix in string seed, then discard an initial batch of 64 values.
      for (var k = 0; k < strseed.length + 20; k++) {
        me.b ^= strseed.charCodeAt(k) | 0;
        me.next();
      }
    }

    function copy(f, t) {
      t.a = f.a;
      t.b = f.b;
      t.c = f.c;
      t.d = f.d;
      return t;
    }
    function impl(seed, opts) {
      var xg = new XorGen(seed),
          state = opts && opts.state,
          prng = function() { return (xg.next() >>> 0) / 0x100000000; };
      prng.double = function() {
        do {
          var top = xg.next() >>> 11,
              bot = (xg.next() >>> 0) / 0x100000000,
              result = (top + bot) / (1 << 21);
        } while (result === 0);
        return result;
      };
      prng.int32 = xg.next;
      prng.quick = prng;
      if (state) {
        if (typeof(state) == 'object') copy(state, xg);
        prng.state = function() { return copy(xg, {}); };
      }
      return prng;
    }

    if (module && module.exports) {
      module.exports = impl;
    } else if (define && define.amd) {
      define(function() { return impl; });
    } else {
      this.tychei = impl;
    }

    })(
      commonjsGlobal,
      module,    // present in node.js
      (typeof undefined) == 'function'   // present with an AMD loader
    );
    });

    var seedrandom = createCommonjsModule(function (module) {
    /*
    Copyright 2014 David Bau.

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    */

    (function (pool, math) {
    //
    // The following constants are related to IEEE 754 limits.
    //
    var global = this,
        width = 256,        // each RC4 output is 0 <= x < 256
        chunks = 6,         // at least six RC4 outputs for each double
        digits = 52,        // there are 52 significant digits in a double
        rngname = 'random', // rngname: name for Math.random and Math.seedrandom
        startdenom = math.pow(width, chunks),
        significance = math.pow(2, digits),
        overflow = significance * 2,
        mask = width - 1,
        nodecrypto;         // node.js crypto module, initialized at the bottom.

    //
    // seedrandom()
    // This is the seedrandom function described above.
    //
    function seedrandom(seed, options, callback) {
      var key = [];
      options = (options == true) ? { entropy: true } : (options || {});

      // Flatten the seed string or build one from local entropy if needed.
      var shortseed = mixkey(flatten(
        options.entropy ? [seed, tostring(pool)] :
        (seed == null) ? autoseed() : seed, 3), key);

      // Use the seed to initialize an ARC4 generator.
      var arc4 = new ARC4(key);

      // This function returns a random double in [0, 1) that contains
      // randomness in every bit of the mantissa of the IEEE 754 value.
      var prng = function() {
        var n = arc4.g(chunks),             // Start with a numerator n < 2 ^ 48
            d = startdenom,                 //   and denominator d = 2 ^ 48.
            x = 0;                          //   and no 'extra last byte'.
        while (n < significance) {          // Fill up all significant digits by
          n = (n + x) * width;              //   shifting numerator and
          d *= width;                       //   denominator and generating a
          x = arc4.g(1);                    //   new least-significant-byte.
        }
        while (n >= overflow) {             // To avoid rounding up, before adding
          n /= 2;                           //   last byte, shift everything
          d /= 2;                           //   right using integer math until
          x >>>= 1;                         //   we have exactly the desired bits.
        }
        return (n + x) / d;                 // Form the number within [0, 1).
      };

      prng.int32 = function() { return arc4.g(4) | 0; };
      prng.quick = function() { return arc4.g(4) / 0x100000000; };
      prng.double = prng;

      // Mix the randomness into accumulated entropy.
      mixkey(tostring(arc4.S), pool);

      // Calling convention: what to return as a function of prng, seed, is_math.
      return (options.pass || callback ||
          function(prng, seed, is_math_call, state) {
            if (state) {
              // Load the arc4 state from the given state if it has an S array.
              if (state.S) { copy(state, arc4); }
              // Only provide the .state method if requested via options.state.
              prng.state = function() { return copy(arc4, {}); };
            }

            // If called as a method of Math (Math.seedrandom()), mutate
            // Math.random because that is how seedrandom.js has worked since v1.0.
            if (is_math_call) { math[rngname] = prng; return seed; }

            // Otherwise, it is a newer calling convention, so return the
            // prng directly.
            else return prng;
          })(
      prng,
      shortseed,
      'global' in options ? options.global : (this == math),
      options.state);
    }
    math['seed' + rngname] = seedrandom;

    //
    // ARC4
    //
    // An ARC4 implementation.  The constructor takes a key in the form of
    // an array of at most (width) integers that should be 0 <= x < (width).
    //
    // The g(count) method returns a pseudorandom integer that concatenates
    // the next (count) outputs from ARC4.  Its return value is a number x
    // that is in the range 0 <= x < (width ^ count).
    //
    function ARC4(key) {
      var t, keylen = key.length,
          me = this, i = 0, j = me.i = me.j = 0, s = me.S = [];

      // The empty key [] is treated as [0].
      if (!keylen) { key = [keylen++]; }

      // Set up S using the standard key scheduling algorithm.
      while (i < width) {
        s[i] = i++;
      }
      for (i = 0; i < width; i++) {
        s[i] = s[j = mask & (j + key[i % keylen] + (t = s[i]))];
        s[j] = t;
      }

      // The "g" method returns the next (count) outputs as one number.
      (me.g = function(count) {
        // Using instance members instead of closure state nearly doubles speed.
        var t, r = 0,
            i = me.i, j = me.j, s = me.S;
        while (count--) {
          t = s[i = mask & (i + 1)];
          r = r * width + s[mask & ((s[i] = s[j = mask & (j + t)]) + (s[j] = t))];
        }
        me.i = i; me.j = j;
        return r;
        // For robust unpredictability, the function call below automatically
        // discards an initial batch of values.  This is called RC4-drop[256].
        // See http://google.com/search?q=rsa+fluhrer+response&btnI
      })(width);
    }

    //
    // copy()
    // Copies internal state of ARC4 to or from a plain object.
    //
    function copy(f, t) {
      t.i = f.i;
      t.j = f.j;
      t.S = f.S.slice();
      return t;
    }
    //
    // flatten()
    // Converts an object tree to nested arrays of strings.
    //
    function flatten(obj, depth) {
      var result = [], typ = (typeof obj), prop;
      if (depth && typ == 'object') {
        for (prop in obj) {
          try { result.push(flatten(obj[prop], depth - 1)); } catch (e) {}
        }
      }
      return (result.length ? result : typ == 'string' ? obj : obj + '\0');
    }

    //
    // mixkey()
    // Mixes a string seed into a key that is an array of integers, and
    // returns a shortened string seed that is equivalent to the result key.
    //
    function mixkey(seed, key) {
      var stringseed = seed + '', smear, j = 0;
      while (j < stringseed.length) {
        key[mask & j] =
          mask & ((smear ^= key[mask & j] * 19) + stringseed.charCodeAt(j++));
      }
      return tostring(key);
    }

    //
    // autoseed()
    // Returns an object for autoseeding, using window.crypto and Node crypto
    // module if available.
    //
    function autoseed() {
      try {
        var out;
        if (nodecrypto && (out = nodecrypto.randomBytes)) {
          // The use of 'out' to remember randomBytes makes tight minified code.
          out = out(width);
        } else {
          out = new Uint8Array(width);
          (global.crypto || global.msCrypto).getRandomValues(out);
        }
        return tostring(out);
      } catch (e) {
        var browser = global.navigator,
            plugins = browser && browser.plugins;
        return [+new Date, global, plugins, global.screen, tostring(pool)];
      }
    }

    //
    // tostring()
    // Converts an array of charcodes to a string
    //
    function tostring(a) {
      return String.fromCharCode.apply(0, a);
    }

    //
    // When seedrandom.js is loaded, we immediately mix a few bits
    // from the built-in RNG into the entropy pool.  Because we do
    // not want to interfere with deterministic PRNG state later,
    // seedrandom will not call math.random on its own again after
    // initialization.
    //
    mixkey(math.random(), pool);

    //
    // Nodejs and AMD support: export the implementation as a module using
    // either convention.
    //
    if (module.exports) {
      module.exports = seedrandom;
      // When in node.js, try using crypto package for autoseeding.
      try {
        nodecrypto = require('crypto');
      } catch (ex) {}
    }

    // End anonymous scope, and pass initial values.
    })(
      [],     // pool: entropy pool starts empty
      Math    // math: package containing random, pow, and seedrandom
    );
    });

    // A library of seedable RNGs implemented in Javascript.
    //
    // Usage:
    //
    // var seedrandom = require('seedrandom');
    // var random = seedrandom(1); // or any seed.
    // var x = random();       // 0 <= x < 1.  Every bit is random.
    // var x = random.quick(); // 0 <= x < 1.  32 bits of randomness.

    // alea, a 53-bit multiply-with-carry generator by Johannes Baagøe.
    // Period: ~2^116
    // Reported to pass all BigCrush tests.


    // xor128, a pure xor-shift generator by George Marsaglia.
    // Period: 2^128-1.
    // Reported to fail: MatrixRank and LinearComp.


    // xorwow, George Marsaglia's 160-bit xor-shift combined plus weyl.
    // Period: 2^192-2^32
    // Reported to fail: CollisionOver, SimpPoker, and LinearComp.


    // xorshift7, by François Panneton and Pierre L'ecuyer, takes
    // a different approach: it adds robustness by allowing more shifts
    // than Marsaglia's original three.  It is a 7-shift generator
    // with 256 bits, that passes BigCrush with no systmatic failures.
    // Period 2^256-1.
    // No systematic BigCrush failures reported.


    // xor4096, by Richard Brent, is a 4096-bit xor-shift with a
    // very long period that also adds a Weyl generator. It also passes
    // BigCrush with no systematic failures.  Its long period may
    // be useful if you have many generators and need to avoid
    // collisions.
    // Period: 2^4128-2^32.
    // No systematic BigCrush failures reported.


    // Tyche-i, by Samuel Neves and Filipe Araujo, is a bit-shifting random
    // number generator derived from ChaCha, a modern stream cipher.
    // https://eden.dei.uc.pt/~sneves/pubs/2011-snfa2.pdf
    // Period: ~2^127
    // No systematic BigCrush failures reported.


    // The original ARC4-based prng included in this library.
    // Period: ~2^1600


    seedrandom.alea = alea;
    seedrandom.xor128 = xor128;
    seedrandom.xorwow = xorwow;
    seedrandom.xorshift7 = xorshift7;
    seedrandom.xor4096 = xor4096;
    seedrandom.tychei = tychei;

    var seedrandom$1 = seedrandom;
    var seedrandom_1 = seedrandom$1.alea;

    var util = createCommonjsModule(function (module, exports) {
    Object.defineProperty(exports, "__esModule", { value: true });
    function shuffle(array) {
        var counter = array.length;
        var temp = 0;
        var index = 0;
        while (counter > 0) {
            index = (Math.random() * counter) | 0;
            counter--;
            temp = array[counter];
            array[counter] = array[index];
            array[index] = temp;
        }
    }
    exports.shuffle = shuffle;
    function clamp(min, x, max) {
        return Math.max(min, Math.min(x, max));
    }
    exports.clamp = clamp;
    function nearestLargerEven(val) {
        return val % 2 === 0 ? val : val + 1;
    }
    exports.nearestLargerEven = nearestLargerEven;
    function randUniform(a, b) {
        var r = Math.random();
        return (b * r) + (1 - r) * a;
    }
    exports.randUniform = randUniform;
    function distSquared(a, b) {
        var result = 0;
        for (var i = 0; i < a.length; i++) {
            var diff = Number(a[i]) - Number(b[i]);
            result += diff * diff;
        }
        return result;
    }
    exports.distSquared = distSquared;
    function assert(expr, msg) {
        if (!expr) {
            throw new Error(typeof msg === 'string' ? msg : msg());
        }
    }
    exports.assert = assert;
    function assertShapesMatch(shapeA, shapeB, errorMessagePrefix) {
        if (errorMessagePrefix === void 0) { errorMessagePrefix = ''; }
        assert(arraysEqual(shapeA, shapeB), errorMessagePrefix + (" Shapes " + shapeA + " and " + shapeB + " must match"));
    }
    exports.assertShapesMatch = assertShapesMatch;
    function assertNonNull(a) {
        assert(a != null, "The input to the tensor constructor must be a non-null value.");
    }
    exports.assertNonNull = assertNonNull;
    function flatten(arr, ret) {
        if (ret === void 0) { ret = []; }
        if (Array.isArray(arr)) {
            for (var i = 0; i < arr.length; ++i) {
                flatten(arr[i], ret);
            }
        }
        else {
            ret.push(arr);
        }
        return ret;
    }
    exports.flatten = flatten;
    function sizeFromShape(shape) {
        if (shape.length === 0) {
            return 1;
        }
        var size = shape[0];
        for (var i = 1; i < shape.length; i++) {
            size *= shape[i];
        }
        return size;
    }
    exports.sizeFromShape = sizeFromShape;
    function isScalarShape(shape) {
        return shape.length === 0;
    }
    exports.isScalarShape = isScalarShape;
    function arraysEqual(n1, n2) {
        if (n1 === n2) {
            return true;
        }
        if (n1 == null || n2 == null) {
            return false;
        }
        if (n1.length !== n2.length) {
            return false;
        }
        for (var i = 0; i < n1.length; i++) {
            if (n1[i] !== n2[i]) {
                return false;
            }
        }
        return true;
    }
    exports.arraysEqual = arraysEqual;
    function isInt(a) {
        return a % 1 === 0;
    }
    exports.isInt = isInt;
    function tanh(x) {
        if (Math.tanh != null) {
            return Math.tanh(x);
        }
        if (x === Infinity) {
            return 1;
        }
        else if (x === -Infinity) {
            return -1;
        }
        else {
            var e2x = Math.exp(2 * x);
            return (e2x - 1) / (e2x + 1);
        }
    }
    exports.tanh = tanh;
    function sizeToSquarishShape(size) {
        for (var a = Math.floor(Math.sqrt(size)); a > 1; --a) {
            if (size % a === 0) {
                return [a, size / a];
            }
        }
        return [1, size];
    }
    exports.sizeToSquarishShape = sizeToSquarishShape;
    function createShuffledIndices(n) {
        var shuffledIndices = new Uint32Array(n);
        for (var i = 0; i < n; ++i) {
            shuffledIndices[i] = i;
        }
        shuffle(shuffledIndices);
        return shuffledIndices;
    }
    exports.createShuffledIndices = createShuffledIndices;
    function rightPad(a, size) {
        if (size <= a.length) {
            return a;
        }
        return a + ' '.repeat(size - a.length);
    }
    exports.rightPad = rightPad;
    function repeatedTry(checkFn, delayFn, maxCounter) {
        if (delayFn === void 0) { delayFn = function (counter) { return 0; }; }
        return new Promise(function (resolve, reject) {
            var tryCount = 0;
            var tryFn = function () {
                if (checkFn()) {
                    resolve();
                    return;
                }
                tryCount++;
                var nextBackoff = delayFn(tryCount);
                if (maxCounter != null && tryCount >= maxCounter) {
                    reject();
                    return;
                }
                setTimeout(tryFn, nextBackoff);
            };
            tryFn();
        });
    }
    exports.repeatedTry = repeatedTry;
    function inferFromImplicitShape(shape, size) {
        var shapeProd = 1;
        var implicitIdx = -1;
        for (var i = 0; i < shape.length; ++i) {
            if (shape[i] >= 0) {
                shapeProd *= shape[i];
            }
            else if (shape[i] === -1) {
                if (implicitIdx !== -1) {
                    throw Error("Shapes can only have 1 implicit size. " +
                        ("Found -1 at dim " + implicitIdx + " and dim " + i));
                }
                implicitIdx = i;
            }
            else if (shape[i] < 0) {
                throw Error("Shapes can not be < 0. Found " + shape[i] + " at dim " + i);
            }
        }
        if (implicitIdx === -1) {
            if (size > 0 && size !== shapeProd) {
                throw Error("Size(" + size + ") must match the product of shape " + shape);
            }
            return shape;
        }
        if (shapeProd === 0) {
            throw Error("Cannot infer the missing size in [" + shape + "] when " +
                "there are 0 elements");
        }
        if (size % shapeProd !== 0) {
            throw Error("The implicit shape can't be a fractional number. " +
                ("Got " + size + " / " + shapeProd));
        }
        var newShape = shape.slice();
        newShape[implicitIdx] = size / shapeProd;
        return newShape;
    }
    exports.inferFromImplicitShape = inferFromImplicitShape;
    function squeezeShape(shape, axis) {
        var newShape = [];
        var keptDims = [];
        var j = 0;
        for (var i = 0; i < shape.length; ++i) {
            if (axis != null) {
                if (axis[j] === i && shape[i] !== 1) {
                    throw new Error("Can't squeeze axis " + i + " since its dim '" + shape[i] + "' is not 1");
                }
                if ((axis[j] == null || axis[j] > i) && shape[i] === 1) {
                    newShape.push(shape[i]);
                    keptDims.push(i);
                }
                if (axis[j] <= i) {
                    j++;
                }
            }
            if (shape[i] !== 1) {
                newShape.push(shape[i]);
                keptDims.push(i);
            }
        }
        return { newShape: newShape, keptDims: keptDims };
    }
    exports.squeezeShape = squeezeShape;
    function getTypedArrayFromDType(dtype, size) {
        var values = null;
        if (dtype == null || dtype === 'float32') {
            values = new Float32Array(size);
        }
        else if (dtype === 'int32') {
            values = new Int32Array(size);
        }
        else if (dtype === 'bool') {
            values = new Uint8Array(size);
        }
        else {
            throw new Error("Unknown data type " + dtype);
        }
        return values;
    }
    exports.getTypedArrayFromDType = getTypedArrayFromDType;
    function checkComputationForNaN(vals, dtype, name) {
        if (dtype !== 'float32') {
            return;
        }
        for (var i = 0; i < vals.length; i++) {
            if (isNaN(vals[i])) {
                throw Error("The result of the '" + name + "' has NaNs.");
            }
        }
    }
    exports.checkComputationForNaN = checkComputationForNaN;
    function checkConversionForNaN(vals, dtype) {
        if (dtype === 'float32') {
            return;
        }
        for (var i = 0; i < vals.length; i++) {
            if (isNaN(vals[i])) {
                throw Error("NaN is not a valid value for dtype: '" + dtype + "'.");
            }
        }
    }
    exports.checkConversionForNaN = checkConversionForNaN;
    function hasEncodingLoss(oldType, newType) {
        if (newType === 'complex64') {
            return false;
        }
        if (newType === 'float32' && oldType !== 'complex64') {
            return false;
        }
        if (newType === 'int32' && oldType !== 'float32' && oldType !== 'complex64') {
            return false;
        }
        if (newType === 'bool' && oldType === 'bool') {
            return false;
        }
        return true;
    }
    exports.hasEncodingLoss = hasEncodingLoss;
    function copyTypedArray(array, dtype, debugMode) {
        if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
            return new Float32Array(array);
        }
        else if (dtype === 'int32') {
            if (debugMode) {
                checkConversionForNaN(array, dtype);
            }
            return new Int32Array(array);
        }
        else if (dtype === 'bool') {
            var bool = new Uint8Array(array.length);
            for (var i = 0; i < bool.length; ++i) {
                if (Math.round(array[i]) !== 0) {
                    bool[i] = 1;
                }
            }
            return bool;
        }
        else {
            throw new Error("Unknown data type " + dtype);
        }
    }
    function isTypedArray(a) {
        return a instanceof Float32Array || a instanceof Int32Array ||
            a instanceof Uint8Array;
    }
    exports.isTypedArray = isTypedArray;
    function bytesPerElement(dtype) {
        if (dtype === 'float32' || dtype === 'int32') {
            return 4;
        }
        else if (dtype === 'complex64') {
            return 8;
        }
        else if (dtype === 'bool') {
            return 1;
        }
        else {
            throw new Error("Unknown dtype " + dtype);
        }
    }
    exports.bytesPerElement = bytesPerElement;
    function isFunction(f) {
        return !!(f && f.constructor && f.call && f.apply);
    }
    exports.isFunction = isFunction;
    function nearestDivisor(size, start) {
        for (var i = start; i < size; ++i) {
            if (size % i === 0) {
                return i;
            }
        }
        return size;
    }
    exports.nearestDivisor = nearestDivisor;
    function computeStrides(shape) {
        var rank = shape.length;
        if (rank < 2) {
            return [];
        }
        var strides = new Array(rank - 1);
        strides[rank - 2] = shape[rank - 1];
        for (var i = rank - 3; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
    exports.computeStrides = computeStrides;
    function toTypedArray(a, dtype, debugMode) {
        if (noConversionNeeded(a, dtype)) {
            return a;
        }
        if (Array.isArray(a)) {
            a = flatten(a);
        }
        return copyTypedArray(a, dtype, debugMode);
    }
    exports.toTypedArray = toTypedArray;
    function noConversionNeeded(a, dtype) {
        return (a instanceof Float32Array && dtype === 'float32') ||
            (a instanceof Int32Array && dtype === 'int32') ||
            (a instanceof Uint8Array && dtype === 'bool');
    }
    function makeOnesTypedArray(size, dtype) {
        var array = makeZerosTypedArray(size, dtype);
        for (var i = 0; i < array.length; i++) {
            array[i] = 1;
        }
        return array;
    }
    exports.makeOnesTypedArray = makeOnesTypedArray;
    function makeZerosTypedArray(size, dtype) {
        if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
            return new Float32Array(size);
        }
        else if (dtype === 'int32') {
            return new Int32Array(size);
        }
        else if (dtype === 'bool') {
            return new Uint8Array(size);
        }
        else {
            throw new Error("Unknown data type " + dtype);
        }
    }
    exports.makeZerosTypedArray = makeZerosTypedArray;
    function now() {
        if (typeof performance !== 'undefined') {
            return performance.now();
        }
        else if (typeof process !== 'undefined') {
            var time = process.hrtime();
            return time[0] * 1000 + time[1] / 1000000;
        }
        else {
            throw new Error('Cannot measure time in this environment. You should run tf.js ' +
                'in the browser or in Node.js');
        }
    }
    exports.now = now;

    });

    unwrapExports(util);
    var util_1 = util.shuffle;
    var util_2 = util.clamp;
    var util_3 = util.nearestLargerEven;
    var util_4 = util.randUniform;
    var util_5 = util.distSquared;
    var util_6 = util.assert;
    var util_7 = util.assertShapesMatch;
    var util_8 = util.assertNonNull;
    var util_9 = util.flatten;
    var util_10 = util.sizeFromShape;
    var util_11 = util.isScalarShape;
    var util_12 = util.arraysEqual;
    var util_13 = util.isInt;
    var util_14 = util.tanh;
    var util_15 = util.sizeToSquarishShape;
    var util_16 = util.createShuffledIndices;
    var util_17 = util.rightPad;
    var util_18 = util.repeatedTry;
    var util_19 = util.inferFromImplicitShape;
    var util_20 = util.squeezeShape;
    var util_21 = util.getTypedArrayFromDType;
    var util_22 = util.checkComputationForNaN;
    var util_23 = util.checkConversionForNaN;
    var util_24 = util.hasEncodingLoss;
    var util_25 = util.isTypedArray;
    var util_26 = util.bytesPerElement;
    var util_27 = util.isFunction;
    var util_28 = util.nearestDivisor;
    var util_29 = util.computeStrides;
    var util_30 = util.toTypedArray;
    var util_31 = util.makeOnesTypedArray;
    var util_32 = util.makeZerosTypedArray;
    var util_33 = util.now;

    var tensor_format = createCommonjsModule(function (module, exports) {
    Object.defineProperty(exports, "__esModule", { value: true });

    var FORMAT_LIMIT_NUM_VALS = 20;
    var FORMAT_NUM_FIRST_LAST_VALS = 3;
    var FORMAT_NUM_SIG_DIGITS = 7;
    function tensorToString(vals, shape, dtype, verbose) {
        var strides = util.computeStrides(shape);
        var padPerCol = computeMaxSizePerColumn(vals, shape, dtype, strides);
        var rank = shape.length;
        var valsLines = subTensorToString(vals, shape, dtype, strides, padPerCol);
        var lines = ['Tensor'];
        if (verbose) {
            lines.push("  dtype: " + dtype);
            lines.push("  rank: " + rank);
            lines.push("  shape: [" + shape + "]");
            lines.push("  values:");
        }
        lines.push(valsLines.map(function (l) { return '    ' + l; }).join('\n'));
        return lines.join('\n');
    }
    exports.tensorToString = tensorToString;
    function computeMaxSizePerColumn(vals, shape, dtype, strides) {
        var n = util.sizeFromShape(shape);
        var numCols = strides[strides.length - 1];
        var padPerCol = new Array(numCols).fill(0);
        var rank = shape.length;
        var valuesOrTuples = dtype === 'complex64' ? createComplexTuples(vals) : vals;
        if (rank > 1) {
            for (var row = 0; row < n / numCols; row++) {
                var offset = row * numCols;
                for (var j = 0; j < numCols; j++) {
                    padPerCol[j] = Math.max(padPerCol[j], valToString(valuesOrTuples[offset + j], 0).length);
                }
            }
        }
        return padPerCol;
    }
    function valToString(val, pad) {
        var valStr;
        if (Array.isArray(val)) {
            valStr = parseFloat(val[0].toFixed(FORMAT_NUM_SIG_DIGITS)) + " + " +
                (parseFloat(val[1].toFixed(FORMAT_NUM_SIG_DIGITS)) + "j");
        }
        else {
            valStr = parseFloat(val.toFixed(FORMAT_NUM_SIG_DIGITS)).toString();
        }
        return util.rightPad(valStr, pad);
    }
    function subTensorToString(vals, shape, dtype, strides, padPerCol, isLast) {
        if (isLast === void 0) { isLast = true; }
        var storagePerElement = dtype === 'complex64' ? 2 : 1;
        var size = shape[0];
        var rank = shape.length;
        if (rank === 0) {
            if (dtype === 'complex64') {
                var complexTuple = createComplexTuples(vals);
                return [valToString(complexTuple[0], 0)];
            }
            return [vals[0].toString()];
        }
        if (rank === 1) {
            if (size > FORMAT_LIMIT_NUM_VALS) {
                var firstValsSize = FORMAT_NUM_FIRST_LAST_VALS * storagePerElement;
                var firstVals = Array.from(vals.subarray(0, firstValsSize));
                var lastVals = Array.from(vals.subarray(size - FORMAT_NUM_FIRST_LAST_VALS * storagePerElement, size));
                if (dtype === 'complex64') {
                    firstVals = createComplexTuples(firstVals);
                    lastVals = createComplexTuples(lastVals);
                }
                return [
                    '[' + firstVals.map(function (x, i) { return valToString(x, padPerCol[i]); }).join(', ') +
                        ', ..., ' +
                        lastVals
                            .map(function (x, i) { return valToString(x, padPerCol[size - FORMAT_NUM_FIRST_LAST_VALS + i]); })
                            .join(', ') +
                        ']'
                ];
            }
            var displayVals = dtype === 'complex64' ? createComplexTuples(vals) : Array.from(vals);
            return [
                '[' + displayVals.map(function (x, i) { return valToString(x, padPerCol[i]); }).join(', ') +
                    ']'
            ];
        }
        var subshape = shape.slice(1);
        var substrides = strides.slice(1);
        var stride = strides[0] * storagePerElement;
        var lines = [];
        if (size > FORMAT_LIMIT_NUM_VALS) {
            for (var i = 0; i < FORMAT_NUM_FIRST_LAST_VALS; i++) {
                var start = i * stride;
                var end = start + stride;
                lines.push.apply(lines, subTensorToString(vals.subarray(start, end), subshape, dtype, substrides, padPerCol, false));
            }
            lines.push('...');
            for (var i = size - FORMAT_NUM_FIRST_LAST_VALS; i < size; i++) {
                var start = i * stride;
                var end = start + stride;
                lines.push.apply(lines, subTensorToString(vals.subarray(start, end), subshape, dtype, substrides, padPerCol, i === size - 1));
            }
        }
        else {
            for (var i = 0; i < size; i++) {
                var start = i * stride;
                var end = start + stride;
                lines.push.apply(lines, subTensorToString(vals.subarray(start, end), subshape, dtype, substrides, padPerCol, i === size - 1));
            }
        }
        var sep = rank === 2 ? ',' : '';
        lines[0] = '[' + lines[0] + sep;
        for (var i = 1; i < lines.length - 1; i++) {
            lines[i] = ' ' + lines[i] + sep;
        }
        var newLineSep = ',\n';
        for (var i = 2; i < rank; i++) {
            newLineSep += '\n';
        }
        lines[lines.length - 1] =
            ' ' + lines[lines.length - 1] + ']' + (isLast ? '' : newLineSep);
        return lines;
    }
    function createComplexTuples(vals) {
        var complexTuples = [];
        for (var i = 0; i < vals.length; i += 2) {
            complexTuples.push([vals[i], vals[i + 1]]);
        }
        return complexTuples;
    }

    });

    unwrapExports(tensor_format);
    var tensor_format_1 = tensor_format.tensorToString;

    var tensor = createCommonjsModule(function (module, exports) {
    var __extends = (commonjsGlobal && commonjsGlobal.__extends) || (function () {
        var extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return function (d, b) {
            extendStatics(d, b);
            function __() { this.constructor = d; }
            d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
        };
    })();
    var __awaiter = (commonjsGlobal && commonjsGlobal.__awaiter) || function (thisArg, _arguments, P, generator) {
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    };
    var __generator = (commonjsGlobal && commonjsGlobal.__generator) || function (thisArg, body) {
        var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
        return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
        function verb(n) { return function (v) { return step([n, v]); }; }
        function step(op) {
            if (f) throw new TypeError("Generator is already executing.");
            while (_) try {
                if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
                if (y = 0, t) op = [op[0] & 2, t.value];
                switch (op[0]) {
                    case 0: case 1: t = op; break;
                    case 4: _.label++; return { value: op[1], done: false };
                    case 5: _.label++; y = op[1]; op = [0]; continue;
                    case 7: op = _.ops.pop(); _.trys.pop(); continue;
                    default:
                        if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                        if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                        if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                        if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                        if (t[2]) _.ops.pop();
                        _.trys.pop(); continue;
                }
                op = body.call(thisArg, _);
            } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
            if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
        }
    };
    Object.defineProperty(exports, "__esModule", { value: true });


    var util_1 = util;
    var TensorBuffer = (function () {
        function TensorBuffer(shape, dtype, values) {
            this.dtype = dtype;
            this.shape = shape.slice();
            this.size = util.sizeFromShape(shape);
            if (values != null) {
                var n = values.length;
                util.assert(n === this.size, "Length of values '" + n + "' does not match the size " +
                    ("inferred by the shape '" + this.size + "'."));
            }
            if (dtype === 'complex64') {
                throw new Error("complex64 dtype TensorBuffers are not supported. Please create " +
                    "a TensorBuffer for the real and imaginary parts separately and " +
                    "call tf.complex(real, imag).");
            }
            this.values = values ||
                util.getTypedArrayFromDType(dtype, util.sizeFromShape(this.shape));
            this.strides = util_1.computeStrides(shape);
        }
        TensorBuffer.prototype.set = function (value) {
            var locs = [];
            for (var _i = 1; _i < arguments.length; _i++) {
                locs[_i - 1] = arguments[_i];
            }
            if (locs.length === 0) {
                locs = [0];
            }
            util.assert(locs.length === this.rank, "The number of provided coordinates (" + locs.length + ") must " +
                ("match the rank (" + this.rank + ")"));
            var index = this.locToIndex(locs);
            this.values[index] = value;
        };
        TensorBuffer.prototype.get = function () {
            var locs = [];
            for (var _i = 0; _i < arguments.length; _i++) {
                locs[_i] = arguments[_i];
            }
            if (locs.length === 0) {
                locs = [0];
            }
            var index = locs[locs.length - 1];
            for (var i = 0; i < locs.length - 1; ++i) {
                index += this.strides[i] * locs[i];
            }
            return this.values[index];
        };
        TensorBuffer.prototype.locToIndex = function (locs) {
            if (this.rank === 0) {
                return 0;
            }
            else if (this.rank === 1) {
                return locs[0];
            }
            var index = locs[locs.length - 1];
            for (var i = 0; i < locs.length - 1; ++i) {
                index += this.strides[i] * locs[i];
            }
            return index;
        };
        TensorBuffer.prototype.indexToLoc = function (index) {
            if (this.rank === 0) {
                return [];
            }
            else if (this.rank === 1) {
                return [index];
            }
            var locs = new Array(this.shape.length);
            for (var i = 0; i < locs.length - 1; ++i) {
                locs[i] = Math.floor(index / this.strides[i]);
                index -= locs[i] * this.strides[i];
            }
            locs[locs.length - 1] = index;
            return locs;
        };
        Object.defineProperty(TensorBuffer.prototype, "rank", {
            get: function () {
                return this.shape.length;
            },
            enumerable: true,
            configurable: true
        });
        TensorBuffer.prototype.toTensor = function () {
            return Tensor.make(this.shape, { values: this.values }, this.dtype);
        };
        return TensorBuffer;
    }());
    exports.TensorBuffer = TensorBuffer;
    var trackerFn = null;
    var opHandler = null;
    function setTensorTracker(fn) {
        trackerFn = fn;
    }
    exports.setTensorTracker = setTensorTracker;
    function setOpHandler(handler) {
        opHandler = handler;
    }
    exports.setOpHandler = setOpHandler;
    var Tensor = (function () {
        function Tensor(shape, dtype, values, dataId) {
            this.isDisposedInternal = false;
            this.shape = shape.slice();
            this.dtype = dtype || 'float32';
            this.size = util.sizeFromShape(shape);
            if (values != null) {
                util.assert(this.size === values.length, "Based on the provided shape, [" + shape + "], and dtype " +
                    (this.dtype + ", the tensor should have ") +
                    (this.size + " values but has " + values.length));
            }
            this.strides = util_1.computeStrides(shape);
            this.dataId = dataId != null ? dataId : {};
            this.id = Tensor.nextId++;
            this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
            trackerFn().registerTensor(this);
            if (values != null) {
                trackerFn().write(this.dataId, values);
            }
        }
        Tensor.make = function (shape, data, dtype) {
            return new Tensor(shape, dtype, data.values, data.dataId);
        };
        Tensor.prototype.flatten = function () {
            this.throwIfDisposed();
            return this.as1D();
        };
        Tensor.prototype.asScalar = function () {
            this.throwIfDisposed();
            util.assert(this.size === 1, 'The array must have only 1 element.');
            return this.reshape([]);
        };
        Tensor.prototype.as1D = function () {
            this.throwIfDisposed();
            return this.reshape([this.size]);
        };
        Tensor.prototype.as2D = function (rows, columns) {
            this.throwIfDisposed();
            return this.reshape([rows, columns]);
        };
        Tensor.prototype.as3D = function (rows, columns, depth) {
            this.throwIfDisposed();
            return this.reshape([rows, columns, depth]);
        };
        Tensor.prototype.as4D = function (rows, columns, depth, depth2) {
            this.throwIfDisposed();
            return this.reshape([rows, columns, depth, depth2]);
        };
        Tensor.prototype.asType = function (dtype) {
            this.throwIfDisposed();
            return opHandler.cast(this, dtype);
        };
        Object.defineProperty(Tensor.prototype, "rank", {
            get: function () {
                return this.shape.length;
            },
            enumerable: true,
            configurable: true
        });
        Tensor.prototype.get = function () {
            var locs = [];
            for (var _i = 0; _i < arguments.length; _i++) {
                locs[_i] = arguments[_i];
            }
            util.assert(locs.length === this.rank, 'Number of coordinates in get() must match the rank of the tensor');
            util.assert(this.dtype !== 'complex64', 'Tensor.get() is not supported for complex64 tensors yet.');
            this.throwIfDisposed();
            if (locs.length === 0) {
                locs = [0];
            }
            var index = locs[locs.length - 1];
            for (var i = 0; i < locs.length - 1; ++i) {
                index += this.strides[i] * locs[i];
            }
            return this.dataSync()[index];
        };
        Tensor.prototype.buffer = function () {
            return opHandler.buffer(this.shape, this.dtype, this.dataSync());
        };
        Tensor.prototype.data = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    this.throwIfDisposed();
                    return [2, trackerFn().read(this.dataId)];
                });
            });
        };
        Tensor.prototype.dataSync = function () {
            this.throwIfDisposed();
            return trackerFn().readSync(this.dataId);
        };
        Tensor.prototype.dispose = function () {
            if (this.isDisposed) {
                return;
            }
            trackerFn().disposeTensor(this);
            this.isDisposedInternal = true;
        };
        Object.defineProperty(Tensor.prototype, "isDisposed", {
            get: function () {
                return this.isDisposedInternal;
            },
            enumerable: true,
            configurable: true
        });
        Tensor.prototype.throwIfDisposed = function () {
            if (this.isDisposed) {
                throw new Error("Tensor is disposed.");
            }
        };
        Tensor.prototype.toFloat = function () {
            return this.asType('float32');
        };
        Tensor.prototype.toInt = function () {
            return this.asType('int32');
        };
        Tensor.prototype.toBool = function () {
            return this.asType('bool');
        };
        Tensor.prototype.print = function (verbose) {
            if (verbose === void 0) { verbose = false; }
            return opHandler.print(this, verbose);
        };
        Tensor.prototype.reshape = function (newShape) {
            this.throwIfDisposed();
            return opHandler.reshape(this, newShape);
        };
        Tensor.prototype.reshapeAs = function (x) {
            this.throwIfDisposed();
            return this.reshape(x.shape);
        };
        Tensor.prototype.expandDims = function (axis) {
            if (axis === void 0) { axis = 0; }
            return opHandler.expandDims(this, axis);
        };
        Tensor.prototype.cumsum = function (axis, exclusive, reverse) {
            if (axis === void 0) { axis = 0; }
            if (exclusive === void 0) { exclusive = false; }
            if (reverse === void 0) { reverse = false; }
            return opHandler.cumsum(this, axis, exclusive, reverse);
        };
        Tensor.prototype.squeeze = function (axis) {
            this.throwIfDisposed();
            return opHandler.squeeze(this, axis);
        };
        Tensor.prototype.clone = function () {
            this.throwIfDisposed();
            return opHandler.clone(this);
        };
        Tensor.prototype.toString = function (verbose) {
            if (verbose === void 0) { verbose = false; }
            var vals = this.dataSync();
            return tensor_format.tensorToString(vals, this.shape, this.dtype, verbose);
        };
        Tensor.prototype.tile = function (reps) {
            this.throwIfDisposed();
            return opHandler.tile(this, reps);
        };
        Tensor.prototype.gather = function (indices, axis) {
            if (axis === void 0) { axis = 0; }
            this.throwIfDisposed();
            return opHandler.gather(this, indices, axis);
        };
        Tensor.prototype.matMul = function (b, transposeA, transposeB) {
            if (transposeA === void 0) { transposeA = false; }
            if (transposeB === void 0) { transposeB = false; }
            this.throwIfDisposed();
            return opHandler.matMul(this, b, transposeA, transposeB);
        };
        Tensor.prototype.dot = function (b) {
            this.throwIfDisposed();
            return opHandler.dot(this, b);
        };
        Tensor.prototype.norm = function (ord, axis, keepDims) {
            if (ord === void 0) { ord = 'euclidean'; }
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.norm(this, ord, axis, keepDims);
        };
        Tensor.prototype.slice = function (begin, size) {
            this.throwIfDisposed();
            return opHandler.slice(this, begin, size);
        };
        Tensor.prototype.reverse = function (axis) {
            this.throwIfDisposed();
            return opHandler.reverse(this, axis);
        };
        Tensor.prototype.concat = function (x, axis) {
            if (axis === void 0) { axis = 0; }
            this.throwIfDisposed();
            return opHandler.concat([this, x], axis);
        };
        Tensor.prototype.split = function (numOrSizeSplits, axis) {
            if (axis === void 0) { axis = 0; }
            this.throwIfDisposed();
            return opHandler.split(this, numOrSizeSplits, axis);
        };
        Tensor.prototype.stack = function (x, axis) {
            if (axis === void 0) { axis = 0; }
            return opHandler.stack([this, x], axis);
        };
        Tensor.prototype.unstack = function (x, axis) {
            if (axis === void 0) { axis = 0; }
            return opHandler.unstack(this, axis);
        };
        Tensor.prototype.pad = function (paddings, constantValue) {
            if (constantValue === void 0) { constantValue = 0; }
            return opHandler.pad(this, paddings, constantValue);
        };
        Tensor.prototype.batchNormalization = function (mean, variance, varianceEpsilon, scale, offset) {
            if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
            this.throwIfDisposed();
            return opHandler.batchNormalization(this, mean, variance, varianceEpsilon, scale, offset);
        };
        Tensor.prototype.all = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.all(this, axis, keepDims);
        };
        Tensor.prototype.any = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.any(this, axis, keepDims);
        };
        Tensor.prototype.logSumExp = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.logSumExp(this, axis, keepDims);
        };
        Tensor.prototype.sum = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.sum(this, axis, keepDims);
        };
        Tensor.prototype.prod = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.prod(this, axis, keepDims);
        };
        Tensor.prototype.mean = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.mean(this, axis, keepDims);
        };
        Tensor.prototype.min = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.min(this, axis, keepDims);
        };
        Tensor.prototype.max = function (axis, keepDims) {
            if (axis === void 0) { axis = null; }
            if (keepDims === void 0) { keepDims = false; }
            this.throwIfDisposed();
            return opHandler.max(this, axis, keepDims);
        };
        Tensor.prototype.argMin = function (axis) {
            if (axis === void 0) { axis = null; }
            this.throwIfDisposed();
            return opHandler.argMin(this, axis);
        };
        Tensor.prototype.argMax = function (axis) {
            if (axis === void 0) { axis = null; }
            this.throwIfDisposed();
            return opHandler.argMax(this, axis);
        };
        Tensor.prototype.cast = function (dtype) {
            this.throwIfDisposed();
            return opHandler.cast(this, dtype);
        };
        Tensor.prototype.add = function (x) {
            this.throwIfDisposed();
            return opHandler.add(this, x);
        };
        Tensor.prototype.addStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.addStrict(this, x);
        };
        Tensor.prototype.atan2 = function (x) {
            this.throwIfDisposed();
            return opHandler.atan2(this, x);
        };
        Tensor.prototype.sub = function (x) {
            this.throwIfDisposed();
            return opHandler.sub(this, x);
        };
        Tensor.prototype.subStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.subStrict(this, x);
        };
        Tensor.prototype.pow = function (exp) {
            this.throwIfDisposed();
            return opHandler.pow(this, exp);
        };
        Tensor.prototype.powStrict = function (exp) {
            this.throwIfDisposed();
            return opHandler.powStrict(this, exp);
        };
        Tensor.prototype.mul = function (x) {
            this.throwIfDisposed();
            return opHandler.mul(this, x);
        };
        Tensor.prototype.mulStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.mulStrict(this, x);
        };
        Tensor.prototype.div = function (x) {
            this.throwIfDisposed();
            return opHandler.div(this, x);
        };
        Tensor.prototype.floorDiv = function (x) {
            this.throwIfDisposed();
            return opHandler.floorDiv(this, x);
        };
        Tensor.prototype.divStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.divStrict(this, x);
        };
        Tensor.prototype.minimum = function (x) {
            this.throwIfDisposed();
            return opHandler.minimum(this, x);
        };
        Tensor.prototype.minimumStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.minimumStrict(this, x);
        };
        Tensor.prototype.maximum = function (x) {
            this.throwIfDisposed();
            return opHandler.maximum(this, x);
        };
        Tensor.prototype.maximumStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.maximumStrict(this, x);
        };
        Tensor.prototype.mod = function (x) {
            this.throwIfDisposed();
            return opHandler.mod(this, x);
        };
        Tensor.prototype.modStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.modStrict(this, x);
        };
        Tensor.prototype.squaredDifference = function (x) {
            this.throwIfDisposed();
            return opHandler.squaredDifference(this, x);
        };
        Tensor.prototype.squaredDifferenceStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.squaredDifferenceStrict(this, x);
        };
        Tensor.prototype.transpose = function (perm) {
            this.throwIfDisposed();
            return opHandler.transpose(this, perm);
        };
        Tensor.prototype.notEqual = function (x) {
            this.throwIfDisposed();
            return opHandler.notEqual(this, x);
        };
        Tensor.prototype.notEqualStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.notEqualStrict(this, x);
        };
        Tensor.prototype.less = function (x) {
            this.throwIfDisposed();
            return opHandler.less(this, x);
        };
        Tensor.prototype.lessStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.lessStrict(this, x);
        };
        Tensor.prototype.equal = function (x) {
            this.throwIfDisposed();
            return opHandler.equal(this, x);
        };
        Tensor.prototype.equalStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.equalStrict(this, x);
        };
        Tensor.prototype.lessEqual = function (x) {
            this.throwIfDisposed();
            return opHandler.lessEqual(this, x);
        };
        Tensor.prototype.lessEqualStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.lessEqualStrict(this, x);
        };
        Tensor.prototype.greater = function (x) {
            this.throwIfDisposed();
            return opHandler.greater(this, x);
        };
        Tensor.prototype.greaterStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.greaterStrict(this, x);
        };
        Tensor.prototype.greaterEqual = function (x) {
            this.throwIfDisposed();
            return opHandler.greaterEqual(this, x);
        };
        Tensor.prototype.greaterEqualStrict = function (x) {
            this.throwIfDisposed();
            return opHandler.greaterEqualStrict(this, x);
        };
        Tensor.prototype.logicalAnd = function (x) {
            this.throwIfDisposed();
            return opHandler.logicalAnd(this, x);
        };
        Tensor.prototype.logicalOr = function (x) {
            this.throwIfDisposed();
            return opHandler.logicalOr(this, x);
        };
        Tensor.prototype.logicalNot = function () {
            this.throwIfDisposed();
            return opHandler.logicalNot(this);
        };
        Tensor.prototype.logicalXor = function (x) {
            this.throwIfDisposed();
            return opHandler.logicalXor(this, x);
        };
        Tensor.prototype.where = function (condition, x) {
            this.throwIfDisposed();
            return opHandler.where(condition, this, x);
        };
        Tensor.prototype.neg = function () {
            this.throwIfDisposed();
            return opHandler.neg(this);
        };
        Tensor.prototype.ceil = function () {
            this.throwIfDisposed();
            return opHandler.ceil(this);
        };
        Tensor.prototype.floor = function () {
            this.throwIfDisposed();
            return opHandler.floor(this);
        };
        Tensor.prototype.sign = function () {
            this.throwIfDisposed();
            return opHandler.sign(this);
        };
        Tensor.prototype.exp = function () {
            this.throwIfDisposed();
            return opHandler.exp(this);
        };
        Tensor.prototype.expm1 = function () {
            this.throwIfDisposed();
            return opHandler.expm1(this);
        };
        Tensor.prototype.log = function () {
            this.throwIfDisposed();
            return opHandler.log(this);
        };
        Tensor.prototype.log1p = function () {
            this.throwIfDisposed();
            return opHandler.log1p(this);
        };
        Tensor.prototype.sqrt = function () {
            this.throwIfDisposed();
            return opHandler.sqrt(this);
        };
        Tensor.prototype.rsqrt = function () {
            this.throwIfDisposed();
            return opHandler.rsqrt(this);
        };
        Tensor.prototype.square = function () {
            this.throwIfDisposed();
            return opHandler.square(this);
        };
        Tensor.prototype.reciprocal = function () {
            this.throwIfDisposed();
            return opHandler.reciprocal(this);
        };
        Tensor.prototype.abs = function () {
            this.throwIfDisposed();
            return opHandler.abs(this);
        };
        Tensor.prototype.clipByValue = function (min, max) {
            this.throwIfDisposed();
            return opHandler.clipByValue(this, min, max);
        };
        Tensor.prototype.relu = function () {
            this.throwIfDisposed();
            return opHandler.relu(this);
        };
        Tensor.prototype.elu = function () {
            this.throwIfDisposed();
            return opHandler.elu(this);
        };
        Tensor.prototype.selu = function () {
            this.throwIfDisposed();
            return opHandler.selu(this);
        };
        Tensor.prototype.leakyRelu = function (alpha) {
            if (alpha === void 0) { alpha = 0.2; }
            this.throwIfDisposed();
            return opHandler.leakyRelu(this, alpha);
        };
        Tensor.prototype.prelu = function (alpha) {
            this.throwIfDisposed();
            return opHandler.prelu(this, alpha);
        };
        Tensor.prototype.sigmoid = function () {
            this.throwIfDisposed();
            return opHandler.sigmoid(this);
        };
        Tensor.prototype.logSigmoid = function () {
            this.throwIfDisposed();
            return opHandler.logSigmoid(this);
        };
        Tensor.prototype.softplus = function () {
            this.throwIfDisposed();
            return opHandler.softplus(this);
        };
        Tensor.prototype.zerosLike = function () {
            this.throwIfDisposed();
            return opHandler.zerosLike(this);
        };
        Tensor.prototype.onesLike = function () {
            this.throwIfDisposed();
            return opHandler.onesLike(this);
        };
        Tensor.prototype.sin = function () {
            this.throwIfDisposed();
            return opHandler.sin(this);
        };
        Tensor.prototype.cos = function () {
            this.throwIfDisposed();
            return opHandler.cos(this);
        };
        Tensor.prototype.tan = function () {
            this.throwIfDisposed();
            return opHandler.tan(this);
        };
        Tensor.prototype.asin = function () {
            this.throwIfDisposed();
            return opHandler.asin(this);
        };
        Tensor.prototype.acos = function () {
            this.throwIfDisposed();
            return opHandler.acos(this);
        };
        Tensor.prototype.atan = function () {
            this.throwIfDisposed();
            return opHandler.atan(this);
        };
        Tensor.prototype.sinh = function () {
            this.throwIfDisposed();
            return opHandler.sinh(this);
        };
        Tensor.prototype.cosh = function () {
            this.throwIfDisposed();
            return opHandler.cosh(this);
        };
        Tensor.prototype.tanh = function () {
            this.throwIfDisposed();
            return opHandler.tanh(this);
        };
        Tensor.prototype.asinh = function () {
            this.throwIfDisposed();
            return opHandler.asinh(this);
        };
        Tensor.prototype.acosh = function () {
            this.throwIfDisposed();
            return opHandler.acosh(this);
        };
        Tensor.prototype.atanh = function () {
            this.throwIfDisposed();
            return opHandler.atanh(this);
        };
        Tensor.prototype.erf = function () {
            this.throwIfDisposed();
            return opHandler.erf(this);
        };
        Tensor.prototype.round = function () {
            this.throwIfDisposed();
            return opHandler.round(this);
        };
        Tensor.prototype.step = function (alpha) {
            if (alpha === void 0) { alpha = 0.0; }
            this.throwIfDisposed();
            return opHandler.step(this, alpha);
        };
        Tensor.prototype.softmax = function (dim) {
            if (dim === void 0) { dim = -1; }
            this.throwIfDisposed();
            return opHandler.softmax(this, dim);
        };
        Tensor.prototype.resizeBilinear = function (newShape2D, alignCorners) {
            if (alignCorners === void 0) { alignCorners = false; }
            this.throwIfDisposed();
            return opHandler.image.resizeBilinear(this, newShape2D, alignCorners);
        };
        Tensor.prototype.resizeNearestNeighbor = function (newShape2D, alignCorners) {
            if (alignCorners === void 0) { alignCorners = false; }
            this.throwIfDisposed();
            return opHandler.image.resizeNearestNeighbor(this, newShape2D, alignCorners);
        };
        Tensor.prototype.conv1d = function (filter, stride, pad, dataFormat, dilation, dimRoundingMode) {
            if (dataFormat === void 0) { dataFormat = 'NWC'; }
            if (dilation === void 0) { dilation = 1; }
            this.throwIfDisposed();
            return opHandler.conv1d(this, filter, stride, pad, dataFormat, dilation, dimRoundingMode);
        };
        Tensor.prototype.conv2d = function (filter, strides, pad, dataFormat, dilations, dimRoundingMode) {
            if (dataFormat === void 0) { dataFormat = 'NHWC'; }
            if (dilations === void 0) { dilations = [1, 1]; }
            this.throwIfDisposed();
            return opHandler.conv2d(this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
        };
        Tensor.prototype.conv2dTranspose = function (filter, outputShape, strides, pad, dimRoundingMode) {
            this.throwIfDisposed();
            return opHandler.conv2dTranspose(this, filter, outputShape, strides, pad, dimRoundingMode);
        };
        Tensor.prototype.depthwiseConv2D = function (filter, strides, pad, dataFormat, dilations, dimRoundingMode) {
            if (dataFormat === void 0) { dataFormat = 'NHWC'; }
            if (dilations === void 0) { dilations = [1, 1]; }
            this.throwIfDisposed();
            return opHandler.depthwiseConv2d(this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
        };
        Tensor.prototype.separableConv2d = function (depthwiseFilter, pointwiseFilter, strides, pad, dilation, dataFormat) {
            if (dilation === void 0) { dilation = [1, 1]; }
            if (dataFormat === void 0) { dataFormat = 'NHWC'; }
            this.throwIfDisposed();
            return opHandler.separableConv2d(this, depthwiseFilter, pointwiseFilter, strides, pad, dilation, dataFormat);
        };
        Tensor.prototype.avgPool = function (filterSize, strides, pad, dimRoundingMode) {
            this.throwIfDisposed();
            return opHandler.avgPool(this, filterSize, strides, pad, dimRoundingMode);
        };
        Tensor.prototype.maxPool = function (filterSize, strides, pad, dimRoundingMode) {
            this.throwIfDisposed();
            return opHandler.maxPool(this, filterSize, strides, pad, dimRoundingMode);
        };
        Tensor.prototype.localResponseNormalization = function (radius, bias, alpha, beta) {
            if (radius === void 0) { radius = 5; }
            if (bias === void 0) { bias = 1; }
            if (alpha === void 0) { alpha = 1; }
            if (beta === void 0) { beta = 0.5; }
            return opHandler.localResponseNormalization(this, radius, bias, alpha, beta);
        };
        Tensor.prototype.variable = function (trainable, name, dtype) {
            if (trainable === void 0) { trainable = true; }
            this.throwIfDisposed();
            return Variable.variable(this, trainable, name, dtype);
        };
        Tensor.prototype.unsortedSegmentSum = function (segmentIds, numSegments) {
            this.throwIfDisposed();
            return opHandler.unsortedSegmentSum(this, segmentIds, numSegments);
        };
        Tensor.prototype.batchToSpaceND = function (blockShape, crops) {
            this.throwIfDisposed();
            return opHandler.batchToSpaceND(this, blockShape, crops);
        };
        Tensor.prototype.spaceToBatchND = function (blockShape, paddings) {
            this.throwIfDisposed();
            return opHandler.spaceToBatchND(this, blockShape, paddings);
        };
        Tensor.prototype.topk = function (k, sorted) {
            if (k === void 0) { k = 1; }
            if (sorted === void 0) { sorted = true; }
            this.throwIfDisposed();
            return opHandler.topk(this, k, sorted);
        };
        Tensor.prototype.stridedSlice = function (begin, end, strides, beginMask, endMask) {
            if (beginMask === void 0) { beginMask = 0; }
            if (endMask === void 0) { endMask = 0; }
            this.throwIfDisposed();
            return opHandler.stridedSlice(this, begin, end, strides, beginMask, endMask);
        };
        Tensor.prototype.depthToSpace = function (blockSize, dataFormat) {
            this.throwIfDisposed();
            return opHandler.depthToSpace(this, blockSize, dataFormat);
        };
        Tensor.prototype.fft = function () {
            this.throwIfDisposed();
            return opHandler.spectral.fft(this);
        };
        Tensor.nextId = 0;
        return Tensor;
    }());
    exports.Tensor = Tensor;
    Object.defineProperty(Tensor, Symbol.hasInstance, {
        value: function (instance) {
            return !!instance && instance.shape != null && instance.dtype != null;
        }
    });
    var Variable = (function (_super) {
        __extends(Variable, _super);
        function Variable(initialValue, trainable, name) {
            if (trainable === void 0) { trainable = true; }
            var _this = _super.call(this, initialValue.shape, initialValue.dtype, null, initialValue.dataId) || this;
            _this.trainable = trainable;
            _this.name = name;
            if (_this.name == null) {
                _this.name = Variable.nextVarId.toString();
                Variable.nextVarId++;
            }
            try {
                trackerFn().registerVariable(_this);
            }
            catch (ex) {
                trackerFn().disposeTensor(_this);
                throw ex;
            }
            return _this;
        }
        Variable.variable = function (initialValue, trainable, name, dtype) {
            if (trainable === void 0) { trainable = true; }
            if (dtype != null && dtype !== initialValue.dtype) {
                initialValue = initialValue.asType(dtype);
            }
            return new Variable(initialValue, trainable, name);
        };
        Variable.prototype.assign = function (newValue) {
            if (newValue.dtype !== this.dtype) {
                throw new Error("dtype of the new value (" + newValue.dtype + ") and " +
                    ("previous value (" + this.dtype + ") must match"));
            }
            if (!util.arraysEqual(newValue.shape, this.shape)) {
                throw new Error("shape of the new value (" + newValue.shape + ") and " +
                    ("previous value (" + this.shape + ") must match"));
            }
            trackerFn().disposeTensor(this);
            this.dataId = newValue.dataId;
            trackerFn().registerTensor(this);
        };
        Variable.nextVarId = 0;
        return Variable;
    }(Tensor));
    exports.Variable = Variable;
    Object.defineProperty(Variable, Symbol.hasInstance, {
        value: function (instance) {
            return instance instanceof Tensor && instance.assign != null &&
                instance.assign instanceof Function;
        }
    });
    var variable = Variable.variable;
    exports.variable = variable;

    });

    unwrapExports(tensor);
    var tensor_1 = tensor.TensorBuffer;
    var tensor_2 = tensor.setTensorTracker;
    var tensor_3 = tensor.setOpHandler;
    var tensor_4 = tensor.Tensor;
    var tensor_5 = tensor.Variable;
    var tensor_6 = tensor.variable;

    var tensor_util = createCommonjsModule(function (module, exports) {
    Object.defineProperty(exports, "__esModule", { value: true });


    function assertTypesMatch(a, b) {
        util.assert(a.dtype === b.dtype, "The dtypes of the first(" + a.dtype + ") and" +
            (" second(" + b.dtype + ") input must match"));
    }
    exports.assertTypesMatch = assertTypesMatch;
    function isTensorInList(tensor$$1, tensorList) {
        for (var i = 0; i < tensorList.length; i++) {
            if (tensorList[i].id === tensor$$1.id) {
                return true;
            }
        }
        return false;
    }
    exports.isTensorInList = isTensorInList;
    function flattenNameArrayMap(nameArrayMap, keys) {
        var xs = [];
        if (nameArrayMap instanceof tensor.Tensor) {
            xs.push(nameArrayMap);
        }
        else {
            var xMap = nameArrayMap;
            for (var i = 0; i < keys.length; i++) {
                xs.push(xMap[keys[i]]);
            }
        }
        return xs;
    }
    exports.flattenNameArrayMap = flattenNameArrayMap;
    function unflattenToNameArrayMap(keys, flatArrays) {
        if (keys.length !== flatArrays.length) {
            throw new Error("Cannot unflatten Tensor[], keys and arrays are not of same length.");
        }
        var result = {};
        for (var i = 0; i < keys.length; i++) {
            result[keys[i]] = flatArrays[i];
        }
        return result;
    }
    exports.unflattenToNameArrayMap = unflattenToNameArrayMap;
    function getTensorsInContainer(result) {
        var list = [];
        var seen = new Set();
        walkTensorContainer(result, list, seen);
        return list;
    }
    exports.getTensorsInContainer = getTensorsInContainer;
    function walkTensorContainer(container, list, seen) {
        if (container == null) {
            return;
        }
        if (container instanceof tensor.Tensor) {
            list.push(container);
            return;
        }
        if (!isIterable(container)) {
            return;
        }
        var iterable = container;
        for (var k in iterable) {
            var val = iterable[k];
            if (!seen.has(val)) {
                seen.add(val);
                walkTensorContainer(val, list, seen);
            }
        }
    }
    function isIterable(obj) {
        return Array.isArray(obj) || typeof obj === 'object';
    }

    });

    unwrapExports(tensor_util);
    var tensor_util_1 = tensor_util.assertTypesMatch;
    var tensor_util_2 = tensor_util.isTensorInList;
    var tensor_util_3 = tensor_util.flattenNameArrayMap;
    var tensor_util_4 = tensor_util.unflattenToNameArrayMap;
    var tensor_util_5 = tensor_util.getTensorsInContainer;

    function deepMapInternal(input, mapFn, seen, containedIn) {
        if (seen === void 0) { seen = new Map(); }
        if (containedIn === void 0) { containedIn = new Set(); }
        if (input == null) {
            return null;
        }
        if (containedIn.has(input)) {
            throw new Error('Circular references are not supported.');
        }
        if (seen.has(input)) {
            return seen.get(input);
        }
        var result = mapFn(input);
        if (result.recurse && result.value !== null) {
            throw new Error('A deep map function may not return both a value and recurse=true.');
        }
        if (!result.recurse) {
            seen.set(input, result.value);
            return result.value;
        }
        else if (isIterable(input)) {
            var mappedIterable = Array.isArray(input) ? [] : {};
            containedIn.add(input);
            for (var k in input) {
                var child = input[k];
                var childResult = deepMapInternal(child, mapFn, seen, containedIn);
                mappedIterable[k] = childResult;
            }
            containedIn.delete(input);
            return mappedIterable;
        }
        else {
            throw new Error("Can't recurse into non-iterable type: " + input);
        }
    }
    function deepZip(inputs, zipFn) {
        if (zipFn === void 0) { zipFn = zipToList; }
        return deepZipInternal(inputs, zipFn);
    }
    function deepZipInternal(inputs, zipFn, containedIn) {
        if (containedIn === void 0) { containedIn = new Set(); }
        var input = inputs[0];
        if (containedIn.has(input)) {
            throw new Error('Circular references are not supported.');
        }
        var result = zipFn(inputs);
        if (result.recurse && result.value !== null) {
            throw new Error('A deep zip function may not return both a value and recurse=true.');
        }
        if (!result.recurse) {
            return result.value;
        }
        else if (isIterable(input)) {
            var mappedIterable = Array.isArray(input) ? [] : {};
            containedIn.add(input);
            var _loop_1 = function (k) {
                var children = inputs.map(function (x) { return x[k]; });
                var childResult = deepZipInternal(children, zipFn, containedIn);
                mappedIterable[k] = childResult;
            };
            for (var k in input) {
                _loop_1(k);
            }
            containedIn.delete(input);
            return mappedIterable;
        }
        else {
            throw new Error("Can't recurse into non-iterable type: " + input);
        }
    }
    function zipToList(x) {
        if (x === null) {
            return null;
        }
        if (isIterable(x[0])) {
            return { value: null, recurse: true };
        }
        else {
            return { value: x, recurse: false };
        }
    }
    function deepMapAndAwaitAll(input, mapFn) {
        return __awaiter(this, void 0, void 0, function () {
            var seen, _i, _a, key, value, mappedValue, result;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        seen = new Map();
                        deepMapInternal(input, mapFn, seen);
                        _i = 0, _a = Array.from(seen.keys());
                        _b.label = 1;
                    case 1:
                        if (!(_i < _a.length)) return [3, 4];
                        key = _a[_i];
                        value = seen.get(key);
                        if (!(value instanceof Promise)) return [3, 3];
                        return [4, value];
                    case 2:
                        mappedValue = _b.sent();
                        seen.set(key, mappedValue);
                        _b.label = 3;
                    case 3:
                        _i++;
                        return [3, 1];
                    case 4:
                        result = deepMapInternal(input, mapFn, seen);
                        return [2, result];
                }
            });
        });
    }
    function isIterable(obj) {
        return obj != null &&
            (Array.isArray(obj) ||
                (typeof obj === 'object' && !(obj instanceof tf.Tensor)));
    }
    function isNumericArray(obj) {
        if (obj == null) {
            return false;
        }
        if (!Array.isArray(obj)) {
            return false;
        }
        for (var k in obj) {
            if (typeof obj[k] !== 'number') {
                return false;
            }
        }
        return true;
    }

    var RingBuffer = (function () {
        function RingBuffer(capacity) {
            this.capacity = capacity;
            this.begin = 0;
            this.end = 0;
            if (capacity < 1) {
                throw new RangeError('Can\'t create ring buffer of capacity < 1.');
            }
            this.data = new Array(capacity);
            this.doubledCapacity = 2 * capacity;
        }
        RingBuffer.prototype.wrap = function (index) {
            while (index < 0) {
                index += this.doubledCapacity;
            }
            return index % this.doubledCapacity;
        };
        RingBuffer.prototype.get = function (index) {
            if (index < 0) {
                throw new RangeError('Can\'t get item at a negative index.');
            }
            return this.data[index % this.capacity];
        };
        RingBuffer.prototype.set = function (index, value) {
            if (index < 0) {
                throw new RangeError('Can\'t set item at a negative index.');
            }
            this.data[index % this.capacity] = value;
        };
        RingBuffer.prototype.length = function () {
            var length = this.end - this.begin;
            if (length < 0) {
                length = this.doubledCapacity + length;
            }
            return length;
        };
        RingBuffer.prototype.isFull = function () {
            return this.length() === this.capacity;
        };
        RingBuffer.prototype.isEmpty = function () {
            return this.length() === 0;
        };
        RingBuffer.prototype.push = function (value) {
            if (this.isFull()) {
                throw new RangeError('Ring buffer is full.');
            }
            this.set(this.end, value);
            this.end = this.wrap(this.end + 1);
        };
        RingBuffer.prototype.pushAll = function (values) {
            for (var _i = 0, values_1 = values; _i < values_1.length; _i++) {
                var value = values_1[_i];
                this.push(value);
            }
        };
        RingBuffer.prototype.pop = function () {
            if (this.isEmpty()) {
                throw new RangeError('Ring buffer is empty.');
            }
            this.end = this.wrap(this.end - 1);
            var result = this.get(this.end);
            this.set(this.end, undefined);
            return result;
        };
        RingBuffer.prototype.unshift = function (value) {
            if (this.isFull()) {
                throw new RangeError('Ring buffer is full.');
            }
            this.begin = this.wrap(this.begin - 1);
            this.set(this.begin, value);
        };
        RingBuffer.prototype.shift = function () {
            if (this.isEmpty()) {
                throw new RangeError('Ring buffer is empty.');
            }
            var result = this.get(this.begin);
            this.set(this.begin, undefined);
            this.begin = this.wrap(this.begin + 1);
            return result;
        };
        RingBuffer.prototype.shuffleExcise = function (relativeIndex) {
            if (this.isEmpty()) {
                throw new RangeError('Ring buffer is empty.');
            }
            var index = this.wrap(this.begin + relativeIndex);
            var result = this.get(index);
            this.set(index, this.pop());
            return result;
        };
        return RingBuffer;
    }());

    var GrowingRingBuffer = (function (_super) {
        __extends(GrowingRingBuffer, _super);
        function GrowingRingBuffer() {
            return _super.call(this, GrowingRingBuffer.INITIAL_CAPACITY) || this;
        }
        GrowingRingBuffer.prototype.isFull = function () {
            return false;
        };
        GrowingRingBuffer.prototype.push = function (value) {
            if (_super.prototype.isFull.call(this)) {
                this.expand();
            }
            _super.prototype.push.call(this, value);
        };
        GrowingRingBuffer.prototype.unshift = function (value) {
            if (_super.prototype.isFull.call(this)) {
                this.expand();
            }
            _super.prototype.unshift.call(this, value);
        };
        GrowingRingBuffer.prototype.expand = function () {
            var newCapacity = this.capacity * 2;
            var newData = new Array(newCapacity);
            var len = this.length();
            for (var i = 0; i < len; i++) {
                newData[i] = this.get(this.wrap(this.begin + i));
            }
            this.data = newData;
            this.capacity = newCapacity;
            this.doubledCapacity = 2 * this.capacity;
            this.begin = 0;
            this.end = len;
        };
        GrowingRingBuffer.INITIAL_CAPACITY = 32;
        return GrowingRingBuffer;
    }(RingBuffer));

    function iteratorFromItems(items) {
        return new ArrayIterator(items);
    }
    function iteratorFromFunction(func) {
        return new FunctionCallIterator(func);
    }
    function iteratorFromConcatenated(baseIterators, baseErrorHandler) {
        return new ChainedIterator(baseIterators, baseErrorHandler);
    }
    function iteratorFromZipped(iterators, mismatchMode) {
        if (mismatchMode === void 0) { mismatchMode = ZipMismatchMode.FAIL; }
        return new ZipIterator(iterators, mismatchMode);
    }
    var LazyIterator = (function () {
        function LazyIterator() {
        }
        LazyIterator.prototype.collect = function (maxItems, prefetch) {
            if (maxItems === void 0) { maxItems = 1000; }
            if (prefetch === void 0) { prefetch = 100; }
            return __awaiter(this, void 0, void 0, function () {
                var stream, result, count, x;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            stream = prefetch > 0 ? this.prefetch(prefetch) : this;
                            result = [];
                            count = 0;
                            return [4, stream.next()];
                        case 1:
                            x = _a.sent();
                            _a.label = 2;
                        case 2:
                            if (!!x.done) return [3, 4];
                            result.push(x.value);
                            count++;
                            if (count >= maxItems) {
                                return [2, result];
                            }
                            return [4, stream.next()];
                        case 3:
                            x = _a.sent();
                            return [3, 2];
                        case 4: return [2, result];
                    }
                });
            });
        };
        LazyIterator.prototype.resolveFully = function () {
            return __awaiter(this, void 0, void 0, function () {
                var x;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.next()];
                        case 1:
                            x = _a.sent();
                            _a.label = 2;
                        case 2:
                            if (!!x.done) return [3, 4];
                            return [4, this.next()];
                        case 3:
                            x = _a.sent();
                            return [3, 2];
                        case 4: return [2];
                    }
                });
            });
        };
        LazyIterator.prototype.resolveWhile = function (predicate) {
            return __awaiter(this, void 0, void 0, function () {
                var x, shouldContinue;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.next()];
                        case 1:
                            x = _a.sent();
                            shouldContinue = predicate(x.value);
                            _a.label = 2;
                        case 2:
                            if (!((!x.done) && shouldContinue)) return [3, 4];
                            return [4, this.next()];
                        case 3:
                            x = _a.sent();
                            shouldContinue = predicate(x.value);
                            return [3, 2];
                        case 4: return [2];
                    }
                });
            });
        };
        LazyIterator.prototype.handleErrors = function (handler) {
            return new ErrorHandlingLazyIterator(this, handler);
        };
        LazyIterator.prototype.filter = function (predicate) {
            return new FilterIterator(this, predicate);
        };
        LazyIterator.prototype.map = function (transform) {
            return new MapIterator(this, transform);
        };
        LazyIterator.prototype.mapAsync = function (transform) {
            return new AsyncMapIterator(this, transform);
        };
        LazyIterator.prototype.serialMapAsync = function (transform) {
            return new AsyncMapIterator(this, transform).serial();
        };
        LazyIterator.prototype.flatmap = function (transform) {
            return new FlatmapIterator(this, transform);
        };
        LazyIterator.prototype.forEach = function (f) {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, this.map(f).resolveFully()];
                });
            });
        };
        LazyIterator.prototype.serialForEach = function (f) {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, this.serialMapAsync(f).resolveWhile(function (x) { return (x === true); })];
                });
            });
        };
        LazyIterator.prototype.rowMajorBatch = function (batchSize, smallLastBatch) {
            if (smallLastBatch === void 0) { smallLastBatch = true; }
            return new RowMajorBatchIterator(this, batchSize, smallLastBatch);
        };
        LazyIterator.prototype.columnMajorBatch = function (batchSize, smallLastBatch, zipFn) {
            if (smallLastBatch === void 0) { smallLastBatch = true; }
            if (zipFn === void 0) { zipFn = zipToList; }
            var rowBatches = this.rowMajorBatch(batchSize, smallLastBatch);
            return rowBatches.map(function (x) { return deepZip(x, zipFn); });
        };
        LazyIterator.prototype.concatenate = function (iterator, baseErrorHandler) {
            return new ChainedIterator(iteratorFromItems([this, iterator]), baseErrorHandler);
        };
        LazyIterator.prototype.take = function (count) {
            if (count < 0 || count == null) {
                return this;
            }
            return new TakeIterator(this, count);
        };
        LazyIterator.prototype.skip = function (count) {
            if (count < 0 || count == null) {
                return this;
            }
            return new SkipIterator(this, count);
        };
        LazyIterator.prototype.prefetch = function (bufferSize) {
            return new PrefetchIterator(this, bufferSize);
        };
        LazyIterator.prototype.shuffle = function (windowSize, seed) {
            return new ShuffleIterator(this, windowSize, seed);
        };
        LazyIterator.prototype.serial = function () {
            return new SerialIterator(this);
        };
        return LazyIterator;
    }());
    var ArrayIterator = (function (_super) {
        __extends(ArrayIterator, _super);
        function ArrayIterator(items) {
            var _this = _super.call(this) || this;
            _this.items = items;
            _this.trav = 0;
            return _this;
        }
        ArrayIterator.prototype.summary = function () {
            return "Array of " + this.items.length + " items";
        };
        ArrayIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var result;
                return __generator(this, function (_a) {
                    if (this.trav >= this.items.length) {
                        return [2, { value: null, done: true }];
                    }
                    result = this.items[this.trav];
                    this.trav++;
                    return [2, { value: result, done: false }];
                });
            });
        };
        return ArrayIterator;
    }(LazyIterator));
    var FunctionCallIterator = (function (_super) {
        __extends(FunctionCallIterator, _super);
        function FunctionCallIterator(nextFn) {
            var _this = _super.call(this) || this;
            _this.nextFn = nextFn;
            return _this;
        }
        FunctionCallIterator.prototype.summary = function () {
            return "Function call";
        };
        FunctionCallIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    try {
                        return [2, this.nextFn()];
                    }
                    catch (e) {
                        e.message =
                            "Error thrown while iterating through a dataset: " + e.message;
                        throw e;
                    }
                    return [2];
                });
            });
        };
        return FunctionCallIterator;
    }(LazyIterator));
    var SerialIterator = (function (_super) {
        __extends(SerialIterator, _super);
        function SerialIterator(upstream) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.lastRead = Promise.resolve({ value: null, done: false });
            return _this;
        }
        SerialIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> Serial";
        };
        SerialIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    this.lastRead = this.lastRead.then(function () { return _this.serialNext(); });
                    return [2, this.lastRead];
                });
            });
        };
        SerialIterator.prototype.serialNext = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, this.upstream.next()];
                });
            });
        };
        return SerialIterator;
    }(LazyIterator));
    var SkipIterator = (function (_super) {
        __extends(SkipIterator, _super);
        function SkipIterator(upstream, maxCount) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.maxCount = maxCount;
            _this.count = 0;
            _this.lastRead = Promise.resolve({ value: null, done: false });
            return _this;
        }
        SkipIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> Skip";
        };
        SkipIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    this.lastRead = this.lastRead.then(function () { return _this.serialNext(); });
                    return [2, this.lastRead];
                });
            });
        };
        SkipIterator.prototype.serialNext = function () {
            return __awaiter(this, void 0, void 0, function () {
                var skipped;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!(this.count++ < this.maxCount)) return [3, 2];
                            return [4, this.upstream.next()];
                        case 1:
                            skipped = _a.sent();
                            if (skipped.done) {
                                return [2, skipped];
                            }
                            tf.dispose(skipped.value);
                            return [3, 0];
                        case 2: return [2, this.upstream.next()];
                    }
                });
            });
        };
        return SkipIterator;
    }(LazyIterator));
    var TakeIterator = (function (_super) {
        __extends(TakeIterator, _super);
        function TakeIterator(upstream, maxCount) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.maxCount = maxCount;
            _this.count = 0;
            return _this;
        }
        TakeIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> Take";
        };
        TakeIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    if (this.count++ >= this.maxCount) {
                        return [2, { value: null, done: true }];
                    }
                    return [2, this.upstream.next()];
                });
            });
        };
        return TakeIterator;
    }(LazyIterator));
    var RowMajorBatchIterator = (function (_super) {
        __extends(RowMajorBatchIterator, _super);
        function RowMajorBatchIterator(upstream, batchSize, enableSmallLastBatch) {
            if (enableSmallLastBatch === void 0) { enableSmallLastBatch = true; }
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.batchSize = batchSize;
            _this.enableSmallLastBatch = enableSmallLastBatch;
            _this.lastRead = Promise.resolve({ value: null, done: false });
            return _this;
        }
        RowMajorBatchIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> RowMajorBatch";
        };
        RowMajorBatchIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    this.lastRead = this.lastRead.then(function () { return _this.serialNext(); });
                    return [2, this.lastRead];
                });
            });
        };
        RowMajorBatchIterator.prototype.serialNext = function () {
            return __awaiter(this, void 0, void 0, function () {
                var batch, item;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            batch = [];
                            _a.label = 1;
                        case 1:
                            if (!(batch.length < this.batchSize)) return [3, 3];
                            return [4, this.upstream.next()];
                        case 2:
                            item = _a.sent();
                            if (item.done) {
                                if (this.enableSmallLastBatch && batch.length > 0) {
                                    return [2, { value: batch, done: false }];
                                }
                                return [2, { value: null, done: true }];
                            }
                            batch.push(item.value);
                            return [3, 1];
                        case 3: return [2, { value: batch, done: false }];
                    }
                });
            });
        };
        return RowMajorBatchIterator;
    }(LazyIterator));
    var FilterIterator = (function (_super) {
        __extends(FilterIterator, _super);
        function FilterIterator(upstream, predicate) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.predicate = predicate;
            _this.lastRead = Promise.resolve({ value: null, done: false });
            return _this;
        }
        FilterIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> Filter";
        };
        FilterIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    this.lastRead = this.lastRead.then(function () { return _this.serialNext(); });
                    return [2, this.lastRead];
                });
            });
        };
        FilterIterator.prototype.serialNext = function () {
            return __awaiter(this, void 0, void 0, function () {
                var item;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            return [4, this.upstream.next()];
                        case 1:
                            item = _a.sent();
                            if (item.done || this.predicate(item.value)) {
                                return [2, item];
                            }
                            tf.dispose(item.value);
                            return [3, 0];
                        case 2: return [2];
                    }
                });
            });
        };
        return FilterIterator;
    }(LazyIterator));
    var MapIterator = (function (_super) {
        __extends(MapIterator, _super);
        function MapIterator(upstream, transform) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.transform = transform;
            return _this;
        }
        MapIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> Map";
        };
        MapIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var item, inputTensors, mapped, outputTensors, _i, inputTensors_1, t;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.upstream.next()];
                        case 1:
                            item = _a.sent();
                            if (item.done) {
                                return [2, { value: null, done: true }];
                            }
                            inputTensors = tensor_util_5(item.value);
                            mapped = this.transform(item.value);
                            outputTensors = tensor_util_5(mapped);
                            for (_i = 0, inputTensors_1 = inputTensors; _i < inputTensors_1.length; _i++) {
                                t = inputTensors_1[_i];
                                if (!tensor_util_2(t, outputTensors)) {
                                    t.dispose();
                                }
                            }
                            return [2, { value: mapped, done: false }];
                    }
                });
            });
        };
        return MapIterator;
    }(LazyIterator));
    var ErrorHandlingLazyIterator = (function (_super) {
        __extends(ErrorHandlingLazyIterator, _super);
        function ErrorHandlingLazyIterator(upstream, handler) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.handler = handler;
            _this.count = 0;
            _this.lastRead = Promise.resolve({ value: null, done: false });
            return _this;
        }
        ErrorHandlingLazyIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> handleErrors";
        };
        ErrorHandlingLazyIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    this.lastRead = this.lastRead.then(function () { return _this.serialNext(); });
                    return [2, this.lastRead];
                });
            });
        };
        ErrorHandlingLazyIterator.prototype.serialNext = function () {
            return __awaiter(this, void 0, void 0, function () {
                var e_1;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            _a.label = 1;
                        case 1:
                            _a.trys.push([1, 3, , 4]);
                            return [4, this.upstream.next()];
                        case 2: return [2, _a.sent()];
                        case 3:
                            e_1 = _a.sent();
                            if (!this.handler(e_1)) {
                                return [2, { value: null, done: true }];
                            }
                            return [3, 4];
                        case 4: return [3, 0];
                        case 5: return [2];
                    }
                });
            });
        };
        return ErrorHandlingLazyIterator;
    }(LazyIterator));
    var AsyncMapIterator = (function (_super) {
        __extends(AsyncMapIterator, _super);
        function AsyncMapIterator(upstream, transform) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.transform = transform;
            return _this;
        }
        AsyncMapIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> AsyncMap";
        };
        AsyncMapIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var item, inputTensors, mapped, outputTensors, _i, inputTensors_2, t;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.upstream.next()];
                        case 1:
                            item = _a.sent();
                            if (item.done) {
                                return [2, { value: null, done: true }];
                            }
                            inputTensors = tensor_util_5(item.value);
                            return [4, this.transform(item.value)];
                        case 2:
                            mapped = _a.sent();
                            outputTensors = tensor_util_5(mapped);
                            for (_i = 0, inputTensors_2 = inputTensors; _i < inputTensors_2.length; _i++) {
                                t = inputTensors_2[_i];
                                if (!tensor_util_2(t, outputTensors)) {
                                    t.dispose();
                                }
                            }
                            return [2, { value: mapped, done: false }];
                    }
                });
            });
        };
        return AsyncMapIterator;
    }(LazyIterator));
    var OneToManyIterator = (function (_super) {
        __extends(OneToManyIterator, _super);
        function OneToManyIterator() {
            var _this = _super.call(this) || this;
            _this.outputQueue = new GrowingRingBuffer();
            _this.lastRead = Promise.resolve({ value: null, done: false });
            return _this;
        }
        OneToManyIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    this.lastRead = this.lastRead.then(function () { return _this.serialNext(); });
                    return [2, this.lastRead];
                });
            });
        };
        OneToManyIterator.prototype.serialNext = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!(this.outputQueue.length() === 0)) return [3, 2];
                            return [4, this.pump()];
                        case 1:
                            if (!(_a.sent())) {
                                return [2, { value: null, done: true }];
                            }
                            return [3, 0];
                        case 2: return [2, { value: this.outputQueue.shift(), done: false }];
                    }
                });
            });
        };
        return OneToManyIterator;
    }(LazyIterator));
    var FlatmapIterator = (function (_super) {
        __extends(FlatmapIterator, _super);
        function FlatmapIterator(upstream, transform) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.transform = transform;
            return _this;
        }
        FlatmapIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> Flatmap";
        };
        FlatmapIterator.prototype.pump = function () {
            return __awaiter(this, void 0, void 0, function () {
                var item, inputTensors, mappedArray, outputTensors, _i, inputTensors_3, t;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.upstream.next()];
                        case 1:
                            item = _a.sent();
                            if (item.done) {
                                return [2, false];
                            }
                            inputTensors = tensor_util_5(item.value);
                            mappedArray = this.transform(item.value);
                            outputTensors = tensor_util_5(mappedArray);
                            this.outputQueue.pushAll(mappedArray);
                            for (_i = 0, inputTensors_3 = inputTensors; _i < inputTensors_3.length; _i++) {
                                t = inputTensors_3[_i];
                                if (!tensor_util_2(t, outputTensors)) {
                                    t.dispose();
                                }
                            }
                            return [2, true];
                    }
                });
            });
        };
        return FlatmapIterator;
    }(OneToManyIterator));
    var ChainedIterator = (function (_super) {
        __extends(ChainedIterator, _super);
        function ChainedIterator(iterators, baseErrorHandler) {
            var _this = _super.call(this) || this;
            _this.baseErrorHandler = baseErrorHandler;
            _this.lastRead = null;
            _this.iterator = null;
            _this.moreIterators = iterators;
            return _this;
        }
        ChainedIterator.prototype.summary = function () {
            var upstreamSummaries = 'TODO: fill in upstream of chained summaries';
            return upstreamSummaries + " -> Chained";
        };
        ChainedIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    this.lastRead = this.readFromChain(this.lastRead);
                    return [2, this.lastRead];
                });
            });
        };
        ChainedIterator.prototype.readFromChain = function (lastRead) {
            return __awaiter(this, void 0, void 0, function () {
                var iteratorResult, itemResult;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, lastRead];
                        case 1:
                            _a.sent();
                            if (!(this.iterator == null)) return [3, 3];
                            return [4, this.moreIterators.next()];
                        case 2:
                            iteratorResult = _a.sent();
                            if (iteratorResult.done) {
                                return [2, { value: null, done: true }];
                            }
                            this.iterator = iteratorResult.value;
                            if (this.baseErrorHandler != null) {
                                this.iterator = this.iterator.handleErrors(this.baseErrorHandler);
                            }
                            _a.label = 3;
                        case 3: return [4, this.iterator.next()];
                        case 4:
                            itemResult = _a.sent();
                            if (itemResult.done) {
                                this.iterator = null;
                                return [2, this.readFromChain(lastRead)];
                            }
                            return [2, itemResult];
                    }
                });
            });
        };
        return ChainedIterator;
    }(LazyIterator));
    var ZipMismatchMode;
    (function (ZipMismatchMode) {
        ZipMismatchMode[ZipMismatchMode["FAIL"] = 0] = "FAIL";
        ZipMismatchMode[ZipMismatchMode["SHORTEST"] = 1] = "SHORTEST";
        ZipMismatchMode[ZipMismatchMode["LONGEST"] = 2] = "LONGEST";
    })(ZipMismatchMode || (ZipMismatchMode = {}));
    var ZipIterator = (function (_super) {
        __extends(ZipIterator, _super);
        function ZipIterator(iterators, mismatchMode) {
            if (mismatchMode === void 0) { mismatchMode = ZipMismatchMode.FAIL; }
            var _this = _super.call(this) || this;
            _this.iterators = iterators;
            _this.mismatchMode = mismatchMode;
            _this.count = 0;
            _this.currentPromise = null;
            return _this;
        }
        ZipIterator.prototype.summary = function () {
            var upstreamSummaries = 'TODO: fill in upstream of zip summaries';
            return "{" + upstreamSummaries + "} -> Zip";
        };
        ZipIterator.prototype.nextState = function (afterState) {
            return __awaiter(this, void 0, void 0, function () {
                function getNext(container) {
                    if (container instanceof LazyIterator) {
                        var result = container.next();
                        return {
                            value: result.then(function (x) {
                                numIterators++;
                                if (x.done) {
                                    iteratorsDone++;
                                }
                                return x.value;
                            }),
                            recurse: false
                        };
                    }
                    else {
                        return { value: null, recurse: true };
                    }
                }
                var numIterators, iteratorsDone, mapped;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, afterState];
                        case 1:
                            _a.sent();
                            numIterators = 0;
                            iteratorsDone = 0;
                            return [4, deepMapAndAwaitAll(this.iterators, getNext)];
                        case 2:
                            mapped = _a.sent();
                            if (numIterators === iteratorsDone) {
                                return [2, { value: null, done: true }];
                            }
                            if (iteratorsDone > 0) {
                                switch (this.mismatchMode) {
                                    case ZipMismatchMode.FAIL:
                                        throw new Error('Zipped streams should have the same length. ' +
                                            ("Mismatched at element " + this.count + "."));
                                    case ZipMismatchMode.SHORTEST:
                                        return [2, { value: null, done: true }];
                                    case ZipMismatchMode.LONGEST:
                                    default:
                                }
                            }
                            this.count++;
                            return [2, { value: mapped, done: false }];
                    }
                });
            });
        };
        ZipIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.currentPromise = this.nextState(this.currentPromise);
                            return [4, this.currentPromise];
                        case 1: return [2, (_a.sent())];
                    }
                });
            });
        };
        return ZipIterator;
    }(LazyIterator));
    var PrefetchIterator = (function (_super) {
        __extends(PrefetchIterator, _super);
        function PrefetchIterator(upstream, bufferSize) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.bufferSize = bufferSize;
            _this.buffer = new RingBuffer(bufferSize);
            return _this;
        }
        PrefetchIterator.prototype.summary = function () {
            return this.upstream.summary() + " -> Prefetch";
        };
        PrefetchIterator.prototype.refill = function () {
            while (!this.buffer.isFull()) {
                var v = this.upstream.next();
                this.buffer.push(v);
            }
        };
        PrefetchIterator.prototype.next = function () {
            this.refill();
            return this.buffer.shift();
        };
        return PrefetchIterator;
    }(LazyIterator));
    var ShuffleIterator = (function (_super) {
        __extends(ShuffleIterator, _super);
        function ShuffleIterator(upstream, windowSize, seed) {
            var _this = _super.call(this, upstream, windowSize) || this;
            _this.upstream = upstream;
            _this.windowSize = windowSize;
            _this.upstreamExhausted = false;
            _this.random = seedrandom_1(seed || tf.util.now().toString());
            _this.lastRead = Promise.resolve({ value: null, done: false });
            return _this;
        }
        ShuffleIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    this.lastRead = this.lastRead.then(function () { return _this.serialNext(); });
                    return [2, this.lastRead];
                });
            });
        };
        ShuffleIterator.prototype.randomInt = function (max) {
            return Math.floor(this.random() * max);
        };
        ShuffleIterator.prototype.chooseIndex = function () {
            return this.randomInt(this.buffer.length());
        };
        ShuffleIterator.prototype.serialNext = function () {
            return __awaiter(this, void 0, void 0, function () {
                var chosenIndex, result;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!this.upstreamExhausted) {
                                this.refill();
                            }
                            _a.label = 1;
                        case 1:
                            if (!!this.buffer.isEmpty()) return [3, 3];
                            chosenIndex = this.chooseIndex();
                            return [4, this.buffer.shuffleExcise(chosenIndex)];
                        case 2:
                            result = _a.sent();
                            if (result.done) {
                                this.upstreamExhausted = true;
                            }
                            else {
                                this.refill();
                                return [2, result];
                            }
                            return [3, 1];
                        case 3: return [2, { value: null, done: true }];
                    }
                });
            });
        };
        return ShuffleIterator;
    }(PrefetchIterator));

    var Dataset = (function () {
        function Dataset() {
        }
        Dataset.prototype.filter = function (filterer) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, base.iterator()];
                        case 1: return [2, (_a.sent()).filter(function (x) { return tf.tidy(function () { return filterer(x); }); })];
                    }
                });
            }); });
        };
        Dataset.prototype.map = function (transform) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, base.iterator()];
                        case 1: return [2, (_a.sent()).map(function (x) { return tf.tidy(function () { return transform(x); }); })];
                    }
                });
            }); });
        };
        Dataset.prototype.mapAsync = function (transform) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, base.iterator()];
                        case 1: return [2, (_a.sent()).mapAsync(transform)];
                    }
                });
            }); });
        };
        Dataset.prototype.batch = function (batchSize, smallLastBatch) {
            var _this = this;
            if (smallLastBatch === void 0) { smallLastBatch = true; }
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, base.iterator()];
                        case 1: return [2, (_a.sent())
                                .columnMajorBatch(batchSize, smallLastBatch, deepBatchConcat)];
                    }
                });
            }); });
        };
        Dataset.prototype.concatenate = function (dataset) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () { var _a, _b; return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0: return [4, base.iterator()];
                    case 1:
                        _b = (_a = (_c.sent())).concatenate;
                        return [4, dataset.iterator()];
                    case 2: return [2, _b.apply(_a, [_c.sent()])];
                }
            }); }); });
        };
        Dataset.prototype.repeat = function (count) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () {
                var _this = this;
                var iteratorIterator;
                return __generator(this, function (_a) {
                    iteratorIterator = iteratorFromFunction(function () { return __awaiter(_this, void 0, void 0, function () { var _a; return __generator(this, function (_b) {
                        switch (_b.label) {
                            case 0:
                                _a = {};
                                return [4, base.iterator()];
                            case 1: return [2, (_a.value = _b.sent(), _a.done = false, _a)];
                        }
                    }); }); });
                    return [2, iteratorFromConcatenated(iteratorIterator.take(count))];
                });
            }); });
        };
        Dataset.prototype.take = function (count) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () { return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, base.iterator()];
                    case 1: return [2, (_a.sent()).take(count)];
                }
            }); }); });
        };
        Dataset.prototype.skip = function (count) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () { return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, base.iterator()];
                    case 1: return [2, (_a.sent()).skip(count)];
                }
            }); }); });
        };
        Dataset.prototype.shuffle = function (bufferSize, seed, reshuffleEachIteration) {
            var _this = this;
            if (reshuffleEachIteration === void 0) { reshuffleEachIteration = true; }
            var base = this;
            var random = seedrandom_1(seed || tf.util.now().toString());
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () {
                var seed2;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            seed2 = random.int32();
                            if (reshuffleEachIteration) {
                                seed2 += random.int32();
                            }
                            return [4, base.iterator()];
                        case 1: return [2, (_a.sent()).shuffle(bufferSize, seed2.toString())];
                    }
                });
            }); });
        };
        Dataset.prototype.prefetch = function (bufferSize) {
            var _this = this;
            var base = this;
            return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () { return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, base.iterator()];
                    case 1: return [2, (_a.sent()).prefetch(bufferSize)];
                }
            }); }); });
        };
        Dataset.prototype.collectAll = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.iterator()];
                        case 1: return [2, (_a.sent()).collect()];
                    }
                });
            });
        };
        Dataset.prototype.forEach = function (f) {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.iterator()];
                        case 1: return [2, (_a.sent()).forEach(f)];
                    }
                });
            });
        };
        return Dataset;
    }());
    function datasetFromIteratorFn(iteratorFn) {
        return new (function (_super) {
            __extends(class_1, _super);
            function class_1() {
                return _super !== null && _super.apply(this, arguments) || this;
            }
            class_1.prototype.iterator = function () {
                return __awaiter(this, void 0, void 0, function () {
                    return __generator(this, function (_a) {
                        return [2, iteratorFn()];
                    });
                });
            };
            return class_1;
        }(Dataset))();
    }
    function array(items) {
        var _this = this;
        return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () { return __generator(this, function (_a) {
            return [2, iteratorFromItems(items)];
        }); }); });
    }
    function zip(datasets) {
        var _this = this;
        if (!isIterable(datasets)) {
            throw new Error('The argument to zip() must be an object or array.');
        }
        return datasetFromIteratorFn(function () { return __awaiter(_this, void 0, void 0, function () {
            var streams;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, deepMapAndAwaitAll(datasets, function (d) {
                            if (d instanceof Dataset) {
                                return { value: d.iterator(), recurse: false };
                            }
                            else if (isIterable(d)) {
                                return { value: null, recurse: true };
                            }
                            else {
                                throw new Error('Leaves of the structure passed to zip() must be Datasets, ' +
                                    'not primitives.');
                            }
                        })];
                    case 1:
                        streams = _a.sent();
                        return [2, iteratorFromZipped(streams, ZipMismatchMode.SHORTEST)];
                }
            });
        }); });
    }
    function deepBatchConcat(rows) {
        if (rows === null) {
            return null;
        }
        var exampleRow = rows[0];
        if (typeof (exampleRow) === 'string') {
            return { value: rows, recurse: false };
        }
        if (!isIterable(exampleRow)) {
            var value = batchConcat(rows);
            return { value: value, recurse: false };
        }
        if (isNumericArray(exampleRow)) {
            var value = batchConcat(rows);
            return { value: value, recurse: false };
        }
        return { value: null, recurse: true };
    }
    function batchConcat(arrays) {
        var elementShape = shapeAndValues(arrays[0])[0];
        var batchShape = [arrays.length].concat(elementShape);
        var resultVals = new Float32Array(batchShape.reduce(function (x, y) { return x * y; }));
        var offset = 0;
        for (var _i = 0, arrays_1 = arrays; _i < arrays_1.length; _i++) {
            var a = arrays_1[_i];
            var _a = shapeAndValues(a), aShape = _a[0], aVals = _a[1];
            if (!tf.util.arraysEqual(aShape, elementShape)) {
                throw new Error('Elements must have the same shape to be batched');
            }
            resultVals.set(aVals, offset);
            offset += aVals.length;
        }
        return tf.Tensor.make(batchShape, { values: resultVals });
    }
    function shapeAndValues(array) {
        if (array instanceof tf.Tensor) {
            return [array.shape, array.dataSync()];
        }
        else if (Array.isArray(array)) {
            return [[array.length], array];
        }
        else {
            return [[], [array]];
        }
    }

    var TextLineDataset = (function (_super) {
        __extends(TextLineDataset, _super);
        function TextLineDataset(input) {
            var _this = _super.call(this) || this;
            _this.input = input;
            return _this;
        }
        TextLineDataset.prototype.iterator = function () {
            return __awaiter(this, void 0, void 0, function () {
                var inputIterator, utf8Iterator, lineIterator;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.input.iterator()];
                        case 1:
                            inputIterator = _a.sent();
                            utf8Iterator = inputIterator.decodeUTF8();
                            lineIterator = utf8Iterator.split('\n');
                            return [2, lineIterator];
                    }
                });
            });
        };
        return TextLineDataset;
    }(Dataset));

    var CODE_QUOTE = '"';
    var STATE_OUT = Symbol('out');
    var STATE_FIELD = Symbol('field');
    var STATE_QUOTE = Symbol('quote');
    var STATE_QUOTE_AFTER_QUOTE = Symbol('quoteafterquote');
    var STATE_WITHIN_QUOTE_IN_QUOTE = Symbol('quoteinquote');
    var CSVDataset = (function (_super) {
        __extends(CSVDataset, _super);
        function CSVDataset(input, csvConfig) {
            var _this = _super.call(this) || this;
            _this.input = input;
            _this.hasHeader = true;
            _this.fullColumnNames = null;
            _this.columnNamesValidated = false;
            _this.columnConfigs = null;
            _this.configuredColumnsOnly = false;
            _this.delimiter = ',';
            _this.base = new TextLineDataset(input);
            if (!csvConfig) {
                csvConfig = {};
            }
            _this.hasHeader = csvConfig.hasHeader === false ? false : true;
            _this.fullColumnNames = csvConfig.columnNames;
            _this.columnConfigs = csvConfig.columnConfigs;
            _this.configuredColumnsOnly = csvConfig.configuredColumnsOnly;
            _this.delimiter = csvConfig.delimiter ? csvConfig.delimiter : ',';
            return _this;
        }
        CSVDataset.prototype.columnNames = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!!this.columnNamesValidated) return [3, 2];
                            return [4, this.setColumnNames()];
                        case 1:
                            _a.sent();
                            _a.label = 2;
                        case 2: return [2, this.configuredColumnsOnly ? Object.keys(this.columnConfigs) :
                                this.fullColumnNames];
                    }
                });
            });
        };
        CSVDataset.prototype.setColumnNames = function () {
            return __awaiter(this, void 0, void 0, function () {
                var columnNamesFromFile, counts, duplicateNames, _i, _a, key, index;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0: return [4, this.maybeReadHeaderLine()];
                        case 1:
                            columnNamesFromFile = _b.sent();
                            if (!this.fullColumnNames && !columnNamesFromFile) {
                                throw new Error('Column names must be provided if there is no header line.');
                            }
                            else if (this.fullColumnNames && columnNamesFromFile) {
                                util_6(columnNamesFromFile.length === this.fullColumnNames.length, 'The length of provided columnNames (' +
                                    this.fullColumnNames.length.toString() +
                                    ') does not match the length of the header line read from ' +
                                    'file (' + columnNamesFromFile.length.toString() + ').');
                            }
                            if (!this.fullColumnNames) {
                                this.fullColumnNames = columnNamesFromFile;
                            }
                            counts = this.fullColumnNames.reduce(function (countAcc, name) {
                                countAcc[name] = (countAcc[name] + 1) || 1;
                                return countAcc;
                            }, {});
                            duplicateNames = Object.keys(counts).filter(function (name) { return (counts[name] > 1); });
                            util_6(duplicateNames.length === 0, 'Duplicate column names found: ' + duplicateNames.toString());
                            if (this.columnConfigs) {
                                for (_i = 0, _a = Object.keys(this.columnConfigs); _i < _a.length; _i++) {
                                    key = _a[_i];
                                    index = this.fullColumnNames.indexOf(key);
                                    if (index === -1) {
                                        throw new Error('The key "' + key +
                                            '" provided in columnConfigs does not match any of the column ' +
                                            'names (' + this.fullColumnNames.toString() + ').');
                                    }
                                }
                            }
                            this.columnNamesValidated = true;
                            return [2];
                    }
                });
            });
        };
        CSVDataset.prototype.maybeReadHeaderLine = function () {
            return __awaiter(this, void 0, void 0, function () {
                var iter, firstElement, firstLine;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!this.hasHeader) return [3, 3];
                            return [4, this.base.iterator()];
                        case 1:
                            iter = _a.sent();
                            return [4, iter.next()];
                        case 2:
                            firstElement = _a.sent();
                            if (firstElement.done) {
                                throw new Error('No data was found for CSV parsing.');
                            }
                            firstLine = firstElement.value;
                            return [2, firstLine.split(this.delimiter)];
                        case 3: return [2, null];
                    }
                });
            });
        };
        CSVDataset.prototype.iterator = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                var lines;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!!this.columnNamesValidated) return [3, 2];
                            return [4, this.setColumnNames()];
                        case 1:
                            _a.sent();
                            _a.label = 2;
                        case 2: return [4, this.base.iterator()];
                        case 3:
                            lines = _a.sent();
                            if (this.hasHeader) {
                                lines = lines.skip(1);
                            }
                            return [2, lines.map(function (x) { return _this.makeDataElement(x); })];
                    }
                });
            });
        };
        CSVDataset.prototype.makeDataElement = function (line) {
            var values = this.parseRow(line);
            var features = {};
            var labels = {};
            for (var i = 0; i < this.fullColumnNames.length; i++) {
                var key = this.fullColumnNames[i];
                var config = this.columnConfigs ? this.columnConfigs[key] : null;
                if (this.configuredColumnsOnly && !config) {
                    continue;
                }
                else {
                    var value = values[i];
                    var parsedValue = null;
                    if (value === '') {
                        if (config && config.default !== undefined) {
                            parsedValue = config.default;
                        }
                        else if (config && (config.required || config.isLabel)) {
                            throw new Error("Required column " + key + " is empty in this line: " + line);
                        }
                        else {
                            parsedValue = undefined;
                        }
                    }
                    else {
                        var valueAsNum = Number(value);
                        if (isNaN(valueAsNum)) {
                            if (config && config.dtype === 'bool') {
                                parsedValue = this.getBoolean(value);
                            }
                            else {
                                parsedValue = value;
                            }
                        }
                        else if (!config || !config.dtype) {
                            parsedValue = valueAsNum;
                        }
                        else {
                            switch (config.dtype) {
                                case 'float32':
                                    parsedValue = valueAsNum;
                                    break;
                                case 'int32':
                                    parsedValue = Math.floor(valueAsNum);
                                    break;
                                case 'bool':
                                    parsedValue = this.getBoolean(value);
                                    break;
                                default:
                                    parsedValue = valueAsNum;
                            }
                        }
                    }
                    (config && config.isLabel) ? labels[key] = parsedValue :
                        features[key] = parsedValue;
                }
            }
            if (Object.keys(labels).length === 0) {
                return features;
            }
            else {
                return [features, labels];
            }
        };
        CSVDataset.prototype.getBoolean = function (value) {
            if (value === '1' || value.toLowerCase() === 'true') {
                return 1;
            }
            else {
                return 0;
            }
        };
        CSVDataset.prototype.parseRow = function (line) {
            var result = [];
            var readOffset = 0;
            var readLength = line.length;
            var currentState = STATE_FIELD;
            for (var i = 0; i < readLength; i++) {
                switch (currentState) {
                    case STATE_OUT:
                        switch (line.charAt(i)) {
                            case CODE_QUOTE:
                                readOffset = i + 1;
                                currentState = STATE_QUOTE;
                                break;
                            case this.delimiter:
                                result.push('');
                                currentState = STATE_OUT;
                                readOffset = i + 1;
                                break;
                            default:
                                currentState = STATE_FIELD;
                                readOffset = i;
                                break;
                        }
                        break;
                    case STATE_FIELD:
                        switch (line.charAt(i)) {
                            case this.delimiter:
                                result.push(line.substring(readOffset, i));
                                currentState = STATE_OUT;
                                readOffset = i + 1;
                                break;
                            default:
                        }
                        break;
                    case STATE_QUOTE:
                        switch (line.charAt(i)) {
                            case CODE_QUOTE:
                                currentState = STATE_QUOTE_AFTER_QUOTE;
                                break;
                            default:
                        }
                        break;
                    case STATE_QUOTE_AFTER_QUOTE:
                        switch (line.charAt(i)) {
                            case this.delimiter:
                                result.push(line.substring(readOffset, i - 1));
                                currentState = STATE_OUT;
                                readOffset = i + 1;
                                break;
                            case CODE_QUOTE:
                                currentState = STATE_QUOTE;
                                break;
                            default:
                                currentState = STATE_WITHIN_QUOTE_IN_QUOTE;
                                break;
                        }
                        break;
                    case STATE_WITHIN_QUOTE_IN_QUOTE:
                        switch (line.charAt(i)) {
                            case CODE_QUOTE:
                                currentState = STATE_QUOTE;
                                break;
                            default:
                        }
                        break;
                    default:
                }
            }
            if (currentState === STATE_QUOTE_AFTER_QUOTE) {
                result.push(line.substring(readOffset, readLength - 1));
            }
            else {
                result.push(line.substring(readOffset));
            }
            return result;
        };
        return CSVDataset;
    }(Dataset));

    var DataSource = (function () {
        function DataSource() {
        }
        return DataSource;
    }());

    // Based on https://github.com/tmpvar/jsdom/blob/aa85b2abf07766ff7bf5c1f6daafb3726f2f2db5/lib/jsdom/living/blob.js
    // (MIT licensed)

    const BUFFER = Symbol('buffer');
    const TYPE = Symbol('type');

    class Blob$1 {
    	constructor() {
    		this[TYPE] = '';

    		const blobParts = arguments[0];
    		const options = arguments[1];

    		const buffers = [];

    		if (blobParts) {
    			const a = blobParts;
    			const length = Number(a.length);
    			for (let i = 0; i < length; i++) {
    				const element = a[i];
    				let buffer;
    				if (element instanceof Buffer) {
    					buffer = element;
    				} else if (ArrayBuffer.isView(element)) {
    					buffer = Buffer.from(element.buffer, element.byteOffset, element.byteLength);
    				} else if (element instanceof ArrayBuffer) {
    					buffer = Buffer.from(element);
    				} else if (element instanceof Blob$1) {
    					buffer = element[BUFFER];
    				} else {
    					buffer = Buffer.from(typeof element === 'string' ? element : String(element));
    				}
    				buffers.push(buffer);
    			}
    		}

    		this[BUFFER] = Buffer.concat(buffers);

    		let type = options && options.type !== undefined && String(options.type).toLowerCase();
    		if (type && !/[^\u0020-\u007E]/.test(type)) {
    			this[TYPE] = type;
    		}
    	}
    	get size() {
    		return this[BUFFER].length;
    	}
    	get type() {
    		return this[TYPE];
    	}
    	slice() {
    		const size = this.size;

    		const start = arguments[0];
    		const end = arguments[1];
    		let relativeStart, relativeEnd;
    		if (start === undefined) {
    			relativeStart = 0;
    		} else if (start < 0) {
    			relativeStart = Math.max(size + start, 0);
    		} else {
    			relativeStart = Math.min(start, size);
    		}
    		if (end === undefined) {
    			relativeEnd = size;
    		} else if (end < 0) {
    			relativeEnd = Math.max(size + end, 0);
    		} else {
    			relativeEnd = Math.min(end, size);
    		}
    		const span = Math.max(relativeEnd - relativeStart, 0);

    		const buffer = this[BUFFER];
    		const slicedBuffer = buffer.slice(relativeStart, relativeStart + span);
    		const blob = new Blob$1([], { type: arguments[2] });
    		blob[BUFFER] = slicedBuffer;
    		return blob;
    	}
    }

    Object.defineProperties(Blob$1.prototype, {
    	size: { enumerable: true },
    	type: { enumerable: true },
    	slice: { enumerable: true }
    });

    Object.defineProperty(Blob$1.prototype, Symbol.toStringTag, {
    	value: 'Blob',
    	writable: false,
    	enumerable: false,
    	configurable: true
    });

    /**
     * fetch-error.js
     *
     * FetchError interface for operational errors
     */

    /**
     * Create FetchError instance
     *
     * @param   String      message      Error message for human
     * @param   String      type         Error type for machine
     * @param   String      systemError  For Node.js system error
     * @return  FetchError
     */
    function FetchError(message, type, systemError) {
      Error.call(this, message);

      this.message = message;
      this.type = type;

      // when err.type is `system`, err.code contains system error code
      if (systemError) {
        this.code = this.errno = systemError.code;
      }

      // hide custom error implementation details from end-users
      Error.captureStackTrace(this, this.constructor);
    }

    FetchError.prototype = Object.create(Error.prototype);
    FetchError.prototype.constructor = FetchError;
    FetchError.prototype.name = 'FetchError';

    /**
     * body.js
     *
     * Body interface provides common methods for Request and Response
     */

    const Stream = require('stream');

    var _require = require('stream');

    const PassThrough = _require.PassThrough;


    let convert;
    try {
    	convert = require('encoding').convert;
    } catch (e) {}

    const INTERNALS = Symbol('Body internals');

    /**
     * Body mixin
     *
     * Ref: https://fetch.spec.whatwg.org/#body
     *
     * @param   Stream  body  Readable stream
     * @param   Object  opts  Response options
     * @return  Void
     */
    function Body(body) {
    	var _this = this;

    	var _ref = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {},
    	    _ref$size = _ref.size;

    	let size = _ref$size === undefined ? 0 : _ref$size;
    	var _ref$timeout = _ref.timeout;
    	let timeout = _ref$timeout === undefined ? 0 : _ref$timeout;

    	if (body == null) {
    		// body is undefined or null
    		body = null;
    	} else if (typeof body === 'string') ; else if (isURLSearchParams(body)) ; else if (body instanceof Blob$1) ; else if (Buffer.isBuffer(body)) ; else if (Object.prototype.toString.call(body) === '[object ArrayBuffer]') ; else if (body instanceof Stream) ; else {
    		// none of the above
    		// coerce to string
    		body = String(body);
    	}
    	this[INTERNALS] = {
    		body,
    		disturbed: false,
    		error: null
    	};
    	this.size = size;
    	this.timeout = timeout;

    	if (body instanceof Stream) {
    		body.on('error', function (err) {
    			_this[INTERNALS].error = new FetchError(`Invalid response body while trying to fetch ${_this.url}: ${err.message}`, 'system', err);
    		});
    	}
    }

    Body.prototype = {
    	get body() {
    		return this[INTERNALS].body;
    	},

    	get bodyUsed() {
    		return this[INTERNALS].disturbed;
    	},

    	/**
      * Decode response as ArrayBuffer
      *
      * @return  Promise
      */
    	arrayBuffer() {
    		return consumeBody.call(this).then(function (buf) {
    			return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    		});
    	},

    	/**
      * Return raw response as Blob
      *
      * @return Promise
      */
    	blob() {
    		let ct = this.headers && this.headers.get('content-type') || '';
    		return consumeBody.call(this).then(function (buf) {
    			return Object.assign(
    			// Prevent copying
    			new Blob$1([], {
    				type: ct.toLowerCase()
    			}), {
    				[BUFFER]: buf
    			});
    		});
    	},

    	/**
      * Decode response as json
      *
      * @return  Promise
      */
    	json() {
    		var _this2 = this;

    		return consumeBody.call(this).then(function (buffer) {
    			try {
    				return JSON.parse(buffer.toString());
    			} catch (err) {
    				return Body.Promise.reject(new FetchError(`invalid json response body at ${_this2.url} reason: ${err.message}`, 'invalid-json'));
    			}
    		});
    	},

    	/**
      * Decode response as text
      *
      * @return  Promise
      */
    	text() {
    		return consumeBody.call(this).then(function (buffer) {
    			return buffer.toString();
    		});
    	},

    	/**
      * Decode response as buffer (non-spec api)
      *
      * @return  Promise
      */
    	buffer() {
    		return consumeBody.call(this);
    	},

    	/**
      * Decode response as text, while automatically detecting the encoding and
      * trying to decode to UTF-8 (non-spec api)
      *
      * @return  Promise
      */
    	textConverted() {
    		var _this3 = this;

    		return consumeBody.call(this).then(function (buffer) {
    			return convertBody(buffer, _this3.headers);
    		});
    	}

    };

    // In browsers, all properties are enumerable.
    Object.defineProperties(Body.prototype, {
    	body: { enumerable: true },
    	bodyUsed: { enumerable: true },
    	arrayBuffer: { enumerable: true },
    	blob: { enumerable: true },
    	json: { enumerable: true },
    	text: { enumerable: true }
    });

    Body.mixIn = function (proto) {
    	for (const name of Object.getOwnPropertyNames(Body.prototype)) {
    		// istanbul ignore else: future proof
    		if (!(name in proto)) {
    			const desc = Object.getOwnPropertyDescriptor(Body.prototype, name);
    			Object.defineProperty(proto, name, desc);
    		}
    	}
    };

    /**
     * Consume and convert an entire Body to a Buffer.
     *
     * Ref: https://fetch.spec.whatwg.org/#concept-body-consume-body
     *
     * @return  Promise
     */
    function consumeBody() {
    	var _this4 = this;

    	if (this[INTERNALS].disturbed) {
    		return Body.Promise.reject(new TypeError(`body used already for: ${this.url}`));
    	}

    	this[INTERNALS].disturbed = true;

    	if (this[INTERNALS].error) {
    		return Body.Promise.reject(this[INTERNALS].error);
    	}

    	// body is null
    	if (this.body === null) {
    		return Body.Promise.resolve(Buffer.alloc(0));
    	}

    	// body is string
    	if (typeof this.body === 'string') {
    		return Body.Promise.resolve(Buffer.from(this.body));
    	}

    	// body is blob
    	if (this.body instanceof Blob$1) {
    		return Body.Promise.resolve(this.body[BUFFER]);
    	}

    	// body is buffer
    	if (Buffer.isBuffer(this.body)) {
    		return Body.Promise.resolve(this.body);
    	}

    	// body is buffer
    	if (Object.prototype.toString.call(this.body) === '[object ArrayBuffer]') {
    		return Body.Promise.resolve(Buffer.from(this.body));
    	}

    	// istanbul ignore if: should never happen
    	if (!(this.body instanceof Stream)) {
    		return Body.Promise.resolve(Buffer.alloc(0));
    	}

    	// body is stream
    	// get ready to actually consume the body
    	let accum = [];
    	let accumBytes = 0;
    	let abort = false;

    	return new Body.Promise(function (resolve, reject) {
    		let resTimeout;

    		// allow timeout on slow response body
    		if (_this4.timeout) {
    			resTimeout = setTimeout(function () {
    				abort = true;
    				reject(new FetchError(`Response timeout while trying to fetch ${_this4.url} (over ${_this4.timeout}ms)`, 'body-timeout'));
    			}, _this4.timeout);
    		}

    		// handle stream error, such as incorrect content-encoding
    		_this4.body.on('error', function (err) {
    			reject(new FetchError(`Invalid response body while trying to fetch ${_this4.url}: ${err.message}`, 'system', err));
    		});

    		_this4.body.on('data', function (chunk) {
    			if (abort || chunk === null) {
    				return;
    			}

    			if (_this4.size && accumBytes + chunk.length > _this4.size) {
    				abort = true;
    				reject(new FetchError(`content size at ${_this4.url} over limit: ${_this4.size}`, 'max-size'));
    				return;
    			}

    			accumBytes += chunk.length;
    			accum.push(chunk);
    		});

    		_this4.body.on('end', function () {
    			if (abort) {
    				return;
    			}

    			clearTimeout(resTimeout);

    			try {
    				resolve(Buffer.concat(accum));
    			} catch (err) {
    				// handle streams that have accumulated too much data (issue #414)
    				reject(new FetchError(`Could not create Buffer from response body for ${_this4.url}: ${err.message}`, 'system', err));
    			}
    		});
    	});
    }

    /**
     * Detect buffer encoding and convert to target encoding
     * ref: http://www.w3.org/TR/2011/WD-html5-20110113/parsing.html#determining-the-character-encoding
     *
     * @param   Buffer  buffer    Incoming buffer
     * @param   String  encoding  Target encoding
     * @return  String
     */
    function convertBody(buffer, headers) {
    	if (typeof convert !== 'function') {
    		throw new Error('The package `encoding` must be installed to use the textConverted() function');
    	}

    	const ct = headers.get('content-type');
    	let charset = 'utf-8';
    	let res, str;

    	// header
    	if (ct) {
    		res = /charset=([^;]*)/i.exec(ct);
    	}

    	// no charset in content type, peek at response body for at most 1024 bytes
    	str = buffer.slice(0, 1024).toString();

    	// html5
    	if (!res && str) {
    		res = /<meta.+?charset=(['"])(.+?)\1/i.exec(str);
    	}

    	// html4
    	if (!res && str) {
    		res = /<meta[\s]+?http-equiv=(['"])content-type\1[\s]+?content=(['"])(.+?)\2/i.exec(str);

    		if (res) {
    			res = /charset=(.*)/i.exec(res.pop());
    		}
    	}

    	// xml
    	if (!res && str) {
    		res = /<\?xml.+?encoding=(['"])(.+?)\1/i.exec(str);
    	}

    	// found charset
    	if (res) {
    		charset = res.pop();

    		// prevent decode issues when sites use incorrect encoding
    		// ref: https://hsivonen.fi/encoding-menu/
    		if (charset === 'gb2312' || charset === 'gbk') {
    			charset = 'gb18030';
    		}
    	}

    	// turn raw buffers into a single utf-8 buffer
    	return convert(buffer, 'UTF-8', charset).toString();
    }

    /**
     * Detect a URLSearchParams object
     * ref: https://github.com/bitinn/node-fetch/issues/296#issuecomment-307598143
     *
     * @param   Object  obj     Object to detect by type or brand
     * @return  String
     */
    function isURLSearchParams(obj) {
    	// Duck-typing as a necessary condition.
    	if (typeof obj !== 'object' || typeof obj.append !== 'function' || typeof obj.delete !== 'function' || typeof obj.get !== 'function' || typeof obj.getAll !== 'function' || typeof obj.has !== 'function' || typeof obj.set !== 'function') {
    		return false;
    	}

    	// Brand-checking and more duck-typing as optional condition.
    	return obj.constructor.name === 'URLSearchParams' || Object.prototype.toString.call(obj) === '[object URLSearchParams]' || typeof obj.sort === 'function';
    }

    /**
     * Clone body given Res/Req instance
     *
     * @param   Mixed  instance  Response or Request instance
     * @return  Mixed
     */
    function clone(instance) {
    	let p1, p2;
    	let body = instance.body;

    	// don't allow cloning a used body
    	if (instance.bodyUsed) {
    		throw new Error('cannot clone body after it is used');
    	}

    	// check that body is a stream and not form-data object
    	// note: we can't clone the form-data object without having it as a dependency
    	if (body instanceof Stream && typeof body.getBoundary !== 'function') {
    		// tee instance body
    		p1 = new PassThrough();
    		p2 = new PassThrough();
    		body.pipe(p1);
    		body.pipe(p2);
    		// set instance body to teed body and return the other teed body
    		instance[INTERNALS].body = p1;
    		body = p2;
    	}

    	return body;
    }

    /**
     * Performs the operation "extract a `Content-Type` value from |object|" as
     * specified in the specification:
     * https://fetch.spec.whatwg.org/#concept-bodyinit-extract
     *
     * This function assumes that instance.body is present.
     *
     * @param   Mixed  instance  Response or Request instance
     */
    function extractContentType(instance) {
    	const body = instance.body;

    	// istanbul ignore if: Currently, because of a guard in Request, body
    	// can never be null. Included here for completeness.

    	if (body === null) {
    		// body is null
    		return null;
    	} else if (typeof body === 'string') {
    		// body is string
    		return 'text/plain;charset=UTF-8';
    	} else if (isURLSearchParams(body)) {
    		// body is a URLSearchParams
    		return 'application/x-www-form-urlencoded;charset=UTF-8';
    	} else if (body instanceof Blob$1) {
    		// body is blob
    		return body.type || null;
    	} else if (Buffer.isBuffer(body)) {
    		// body is buffer
    		return null;
    	} else if (Object.prototype.toString.call(body) === '[object ArrayBuffer]') {
    		// body is array buffer
    		return null;
    	} else if (typeof body.getBoundary === 'function') {
    		// detect form data input from form-data module
    		return `multipart/form-data;boundary=${body.getBoundary()}`;
    	} else {
    		// body is stream
    		// can't really do much about this
    		return null;
    	}
    }

    /**
     * The Fetch Standard treats this as if "total bytes" is a property on the body.
     * For us, we have to explicitly get it with a function.
     *
     * ref: https://fetch.spec.whatwg.org/#concept-body-total-bytes
     *
     * @param   Body    instance   Instance of Body
     * @return  Number?            Number of bytes, or null if not possible
     */
    function getTotalBytes(instance) {
    	const body = instance.body;

    	// istanbul ignore if: included for completion

    	if (body === null) {
    		// body is null
    		return 0;
    	} else if (typeof body === 'string') {
    		// body is string
    		return Buffer.byteLength(body);
    	} else if (isURLSearchParams(body)) {
    		// body is URLSearchParams
    		return Buffer.byteLength(String(body));
    	} else if (body instanceof Blob$1) {
    		// body is blob
    		return body.size;
    	} else if (Buffer.isBuffer(body)) {
    		// body is buffer
    		return body.length;
    	} else if (Object.prototype.toString.call(body) === '[object ArrayBuffer]') {
    		// body is array buffer
    		return body.byteLength;
    	} else if (body && typeof body.getLengthSync === 'function') {
    		// detect form data input from form-data module
    		if (body._lengthRetrievers && body._lengthRetrievers.length == 0 || // 1.x
    		body.hasKnownLength && body.hasKnownLength()) {
    			// 2.x
    			return body.getLengthSync();
    		}
    		return null;
    	} else {
    		// body is stream
    		// can't really do much about this
    		return null;
    	}
    }

    /**
     * Write a Body to a Node.js WritableStream (e.g. http.Request) object.
     *
     * @param   Body    instance   Instance of Body
     * @return  Void
     */
    function writeToStream(dest, instance) {
    	const body = instance.body;


    	if (body === null) {
    		// body is null
    		dest.end();
    	} else if (typeof body === 'string') {
    		// body is string
    		dest.write(body);
    		dest.end();
    	} else if (isURLSearchParams(body)) {
    		// body is URLSearchParams
    		dest.write(Buffer.from(String(body)));
    		dest.end();
    	} else if (body instanceof Blob$1) {
    		// body is blob
    		dest.write(body[BUFFER]);
    		dest.end();
    	} else if (Buffer.isBuffer(body)) {
    		// body is buffer
    		dest.write(body);
    		dest.end();
    	} else if (Object.prototype.toString.call(body) === '[object ArrayBuffer]') {
    		// body is array buffer
    		dest.write(Buffer.from(body));
    		dest.end();
    	} else {
    		// body is stream
    		body.pipe(dest);
    	}
    }

    // expose Promise
    Body.Promise = global.Promise;

    /**
     * headers.js
     *
     * Headers class offers convenient helpers
     */

    const invalidTokenRegex = /[^\^_`a-zA-Z\-0-9!#$%&'*+.|~]/;
    const invalidHeaderCharRegex = /[^\t\x20-\x7e\x80-\xff]/;

    function validateName(name) {
    	name = `${name}`;
    	if (invalidTokenRegex.test(name)) {
    		throw new TypeError(`${name} is not a legal HTTP header name`);
    	}
    }

    function validateValue(value) {
    	value = `${value}`;
    	if (invalidHeaderCharRegex.test(value)) {
    		throw new TypeError(`${value} is not a legal HTTP header value`);
    	}
    }

    /**
     * Find the key in the map object given a header name.
     *
     * Returns undefined if not found.
     *
     * @param   String  name  Header name
     * @return  String|Undefined
     */
    function find(map, name) {
    	name = name.toLowerCase();
    	for (const key in map) {
    		if (key.toLowerCase() === name) {
    			return key;
    		}
    	}
    	return undefined;
    }

    const MAP = Symbol('map');
    class Headers {
    	/**
      * Headers class
      *
      * @param   Object  headers  Response headers
      * @return  Void
      */
    	constructor() {
    		let init = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : undefined;

    		this[MAP] = Object.create(null);

    		if (init instanceof Headers) {
    			const rawHeaders = init.raw();
    			const headerNames = Object.keys(rawHeaders);

    			for (const headerName of headerNames) {
    				for (const value of rawHeaders[headerName]) {
    					this.append(headerName, value);
    				}
    			}

    			return;
    		}

    		// We don't worry about converting prop to ByteString here as append()
    		// will handle it.
    		if (init == null) ; else if (typeof init === 'object') {
    			const method = init[Symbol.iterator];
    			if (method != null) {
    				if (typeof method !== 'function') {
    					throw new TypeError('Header pairs must be iterable');
    				}

    				// sequence<sequence<ByteString>>
    				// Note: per spec we have to first exhaust the lists then process them
    				const pairs = [];
    				for (const pair of init) {
    					if (typeof pair !== 'object' || typeof pair[Symbol.iterator] !== 'function') {
    						throw new TypeError('Each header pair must be iterable');
    					}
    					pairs.push(Array.from(pair));
    				}

    				for (const pair of pairs) {
    					if (pair.length !== 2) {
    						throw new TypeError('Each header pair must be a name/value tuple');
    					}
    					this.append(pair[0], pair[1]);
    				}
    			} else {
    				// record<ByteString, ByteString>
    				for (const key of Object.keys(init)) {
    					const value = init[key];
    					this.append(key, value);
    				}
    			}
    		} else {
    			throw new TypeError('Provided initializer must be an object');
    		}
    	}

    	/**
      * Return combined header value given name
      *
      * @param   String  name  Header name
      * @return  Mixed
      */
    	get(name) {
    		name = `${name}`;
    		validateName(name);
    		const key = find(this[MAP], name);
    		if (key === undefined) {
    			return null;
    		}

    		return this[MAP][key].join(', ');
    	}

    	/**
      * Iterate over all headers
      *
      * @param   Function  callback  Executed for each item with parameters (value, name, thisArg)
      * @param   Boolean   thisArg   `this` context for callback function
      * @return  Void
      */
    	forEach(callback) {
    		let thisArg = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : undefined;

    		let pairs = getHeaders(this);
    		let i = 0;
    		while (i < pairs.length) {
    			var _pairs$i = pairs[i];
    			const name = _pairs$i[0],
    			      value = _pairs$i[1];

    			callback.call(thisArg, value, name, this);
    			pairs = getHeaders(this);
    			i++;
    		}
    	}

    	/**
      * Overwrite header values given name
      *
      * @param   String  name   Header name
      * @param   String  value  Header value
      * @return  Void
      */
    	set(name, value) {
    		name = `${name}`;
    		value = `${value}`;
    		validateName(name);
    		validateValue(value);
    		const key = find(this[MAP], name);
    		this[MAP][key !== undefined ? key : name] = [value];
    	}

    	/**
      * Append a value onto existing header
      *
      * @param   String  name   Header name
      * @param   String  value  Header value
      * @return  Void
      */
    	append(name, value) {
    		name = `${name}`;
    		value = `${value}`;
    		validateName(name);
    		validateValue(value);
    		const key = find(this[MAP], name);
    		if (key !== undefined) {
    			this[MAP][key].push(value);
    		} else {
    			this[MAP][name] = [value];
    		}
    	}

    	/**
      * Check for header name existence
      *
      * @param   String   name  Header name
      * @return  Boolean
      */
    	has(name) {
    		name = `${name}`;
    		validateName(name);
    		return find(this[MAP], name) !== undefined;
    	}

    	/**
      * Delete all header values given name
      *
      * @param   String  name  Header name
      * @return  Void
      */
    	delete(name) {
    		name = `${name}`;
    		validateName(name);
    		const key = find(this[MAP], name);
    		if (key !== undefined) {
    			delete this[MAP][key];
    		}
    	}

    	/**
      * Return raw headers (non-spec api)
      *
      * @return  Object
      */
    	raw() {
    		return this[MAP];
    	}

    	/**
      * Get an iterator on keys.
      *
      * @return  Iterator
      */
    	keys() {
    		return createHeadersIterator(this, 'key');
    	}

    	/**
      * Get an iterator on values.
      *
      * @return  Iterator
      */
    	values() {
    		return createHeadersIterator(this, 'value');
    	}

    	/**
      * Get an iterator on entries.
      *
      * This is the default iterator of the Headers object.
      *
      * @return  Iterator
      */
    	[Symbol.iterator]() {
    		return createHeadersIterator(this, 'key+value');
    	}
    }
    Headers.prototype.entries = Headers.prototype[Symbol.iterator];

    Object.defineProperty(Headers.prototype, Symbol.toStringTag, {
    	value: 'Headers',
    	writable: false,
    	enumerable: false,
    	configurable: true
    });

    Object.defineProperties(Headers.prototype, {
    	get: { enumerable: true },
    	forEach: { enumerable: true },
    	set: { enumerable: true },
    	append: { enumerable: true },
    	has: { enumerable: true },
    	delete: { enumerable: true },
    	keys: { enumerable: true },
    	values: { enumerable: true },
    	entries: { enumerable: true }
    });

    function getHeaders(headers) {
    	let kind = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'key+value';

    	const keys = Object.keys(headers[MAP]).sort();
    	return keys.map(kind === 'key' ? function (k) {
    		return k.toLowerCase();
    	} : kind === 'value' ? function (k) {
    		return headers[MAP][k].join(', ');
    	} : function (k) {
    		return [k.toLowerCase(), headers[MAP][k].join(', ')];
    	});
    }

    const INTERNAL = Symbol('internal');

    function createHeadersIterator(target, kind) {
    	const iterator = Object.create(HeadersIteratorPrototype);
    	iterator[INTERNAL] = {
    		target,
    		kind,
    		index: 0
    	};
    	return iterator;
    }

    const HeadersIteratorPrototype = Object.setPrototypeOf({
    	next() {
    		// istanbul ignore if
    		if (!this || Object.getPrototypeOf(this) !== HeadersIteratorPrototype) {
    			throw new TypeError('Value of `this` is not a HeadersIterator');
    		}

    		var _INTERNAL = this[INTERNAL];
    		const target = _INTERNAL.target,
    		      kind = _INTERNAL.kind,
    		      index = _INTERNAL.index;

    		const values = getHeaders(target, kind);
    		const len = values.length;
    		if (index >= len) {
    			return {
    				value: undefined,
    				done: true
    			};
    		}

    		this[INTERNAL].index = index + 1;

    		return {
    			value: values[index],
    			done: false
    		};
    	}
    }, Object.getPrototypeOf(Object.getPrototypeOf([][Symbol.iterator]())));

    Object.defineProperty(HeadersIteratorPrototype, Symbol.toStringTag, {
    	value: 'HeadersIterator',
    	writable: false,
    	enumerable: false,
    	configurable: true
    });

    /**
     * Export the Headers object in a form that Node.js can consume.
     *
     * @param   Headers  headers
     * @return  Object
     */
    function exportNodeCompatibleHeaders(headers) {
    	const obj = Object.assign({ __proto__: null }, headers[MAP]);

    	// http.request() only supports string as Host header. This hack makes
    	// specifying custom Host header possible.
    	const hostHeaderKey = find(headers[MAP], 'Host');
    	if (hostHeaderKey !== undefined) {
    		obj[hostHeaderKey] = obj[hostHeaderKey][0];
    	}

    	return obj;
    }

    /**
     * Create a Headers object from an object of headers, ignoring those that do
     * not conform to HTTP grammar productions.
     *
     * @param   Object  obj  Object of headers
     * @return  Headers
     */
    function createHeadersLenient(obj) {
    	const headers = new Headers();
    	for (const name of Object.keys(obj)) {
    		if (invalidTokenRegex.test(name)) {
    			continue;
    		}
    		if (Array.isArray(obj[name])) {
    			for (const val of obj[name]) {
    				if (invalidHeaderCharRegex.test(val)) {
    					continue;
    				}
    				if (headers[MAP][name] === undefined) {
    					headers[MAP][name] = [val];
    				} else {
    					headers[MAP][name].push(val);
    				}
    			}
    		} else if (!invalidHeaderCharRegex.test(obj[name])) {
    			headers[MAP][name] = [obj[name]];
    		}
    	}
    	return headers;
    }

    /**
     * response.js
     *
     * Response class provides content decoding
     */

    var _require$1 = require('http');

    const STATUS_CODES = _require$1.STATUS_CODES;


    const INTERNALS$1 = Symbol('Response internals');

    /**
     * Response class
     *
     * @param   Stream  body  Readable stream
     * @param   Object  opts  Response options
     * @return  Void
     */
    class Response {
    	constructor() {
    		let body = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : null;
    		let opts = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

    		Body.call(this, body, opts);

    		const status = opts.status || 200;

    		this[INTERNALS$1] = {
    			url: opts.url,
    			status,
    			statusText: opts.statusText || STATUS_CODES[status],
    			headers: new Headers(opts.headers)
    		};
    	}

    	get url() {
    		return this[INTERNALS$1].url;
    	}

    	get status() {
    		return this[INTERNALS$1].status;
    	}

    	/**
      * Convenience property representing if the request ended normally
      */
    	get ok() {
    		return this[INTERNALS$1].status >= 200 && this[INTERNALS$1].status < 300;
    	}

    	get statusText() {
    		return this[INTERNALS$1].statusText;
    	}

    	get headers() {
    		return this[INTERNALS$1].headers;
    	}

    	/**
      * Clone this response
      *
      * @return  Response
      */
    	clone() {
    		return new Response(clone(this), {
    			url: this.url,
    			status: this.status,
    			statusText: this.statusText,
    			headers: this.headers,
    			ok: this.ok
    		});
    	}
    }

    Body.mixIn(Response.prototype);

    Object.defineProperties(Response.prototype, {
    	url: { enumerable: true },
    	status: { enumerable: true },
    	ok: { enumerable: true },
    	statusText: { enumerable: true },
    	headers: { enumerable: true },
    	clone: { enumerable: true }
    });

    Object.defineProperty(Response.prototype, Symbol.toStringTag, {
    	value: 'Response',
    	writable: false,
    	enumerable: false,
    	configurable: true
    });

    /**
     * request.js
     *
     * Request class contains server only options
     *
     * All spec algorithm step numbers are based on https://fetch.spec.whatwg.org/commit-snapshots/ae716822cb3a61843226cd090eefc6589446c1d2/.
     */

    var _require$2 = require('url');

    const format_url = _require$2.format;
    const parse_url = _require$2.parse;


    const INTERNALS$2 = Symbol('Request internals');

    /**
     * Check if a value is an instance of Request.
     *
     * @param   Mixed   input
     * @return  Boolean
     */
    function isRequest(input) {
    	return typeof input === 'object' && typeof input[INTERNALS$2] === 'object';
    }

    /**
     * Request class
     *
     * @param   Mixed   input  Url or Request instance
     * @param   Object  init   Custom options
     * @return  Void
     */
    class Request {
    	constructor(input) {
    		let init = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

    		let parsedURL;

    		// normalize input
    		if (!isRequest(input)) {
    			if (input && input.href) {
    				// in order to support Node.js' Url objects; though WHATWG's URL objects
    				// will fall into this branch also (since their `toString()` will return
    				// `href` property anyway)
    				parsedURL = parse_url(input.href);
    			} else {
    				// coerce input to a string before attempting to parse
    				parsedURL = parse_url(`${input}`);
    			}
    			input = {};
    		} else {
    			parsedURL = parse_url(input.url);
    		}

    		let method = init.method || input.method || 'GET';
    		method = method.toUpperCase();

    		if ((init.body != null || isRequest(input) && input.body !== null) && (method === 'GET' || method === 'HEAD')) {
    			throw new TypeError('Request with GET/HEAD method cannot have body');
    		}

    		let inputBody = init.body != null ? init.body : isRequest(input) && input.body !== null ? clone(input) : null;

    		Body.call(this, inputBody, {
    			timeout: init.timeout || input.timeout || 0,
    			size: init.size || input.size || 0
    		});

    		const headers = new Headers(init.headers || input.headers || {});

    		if (init.body != null) {
    			const contentType = extractContentType(this);
    			if (contentType !== null && !headers.has('Content-Type')) {
    				headers.append('Content-Type', contentType);
    			}
    		}

    		this[INTERNALS$2] = {
    			method,
    			redirect: init.redirect || input.redirect || 'follow',
    			headers,
    			parsedURL
    		};

    		// node-fetch-only options
    		this.follow = init.follow !== undefined ? init.follow : input.follow !== undefined ? input.follow : 20;
    		this.compress = init.compress !== undefined ? init.compress : input.compress !== undefined ? input.compress : true;
    		this.counter = init.counter || input.counter || 0;
    		this.agent = init.agent || input.agent;
    	}

    	get method() {
    		return this[INTERNALS$2].method;
    	}

    	get url() {
    		return format_url(this[INTERNALS$2].parsedURL);
    	}

    	get headers() {
    		return this[INTERNALS$2].headers;
    	}

    	get redirect() {
    		return this[INTERNALS$2].redirect;
    	}

    	/**
      * Clone this request
      *
      * @return  Request
      */
    	clone() {
    		return new Request(this);
    	}
    }

    Body.mixIn(Request.prototype);

    Object.defineProperty(Request.prototype, Symbol.toStringTag, {
    	value: 'Request',
    	writable: false,
    	enumerable: false,
    	configurable: true
    });

    Object.defineProperties(Request.prototype, {
    	method: { enumerable: true },
    	url: { enumerable: true },
    	headers: { enumerable: true },
    	redirect: { enumerable: true },
    	clone: { enumerable: true }
    });

    /**
     * Convert a Request to Node.js http request options.
     *
     * @param   Request  A Request instance
     * @return  Object   The options object to be passed to http.request
     */
    function getNodeRequestOptions(request) {
    	const parsedURL = request[INTERNALS$2].parsedURL;
    	const headers = new Headers(request[INTERNALS$2].headers);

    	// fetch step 1.3
    	if (!headers.has('Accept')) {
    		headers.set('Accept', '*/*');
    	}

    	// Basic fetch
    	if (!parsedURL.protocol || !parsedURL.hostname) {
    		throw new TypeError('Only absolute URLs are supported');
    	}

    	if (!/^https?:$/.test(parsedURL.protocol)) {
    		throw new TypeError('Only HTTP(S) protocols are supported');
    	}

    	// HTTP-network-or-cache fetch steps 2.4-2.7
    	let contentLengthValue = null;
    	if (request.body == null && /^(POST|PUT)$/i.test(request.method)) {
    		contentLengthValue = '0';
    	}
    	if (request.body != null) {
    		const totalBytes = getTotalBytes(request);
    		if (typeof totalBytes === 'number') {
    			contentLengthValue = String(totalBytes);
    		}
    	}
    	if (contentLengthValue) {
    		headers.set('Content-Length', contentLengthValue);
    	}

    	// HTTP-network-or-cache fetch step 2.11
    	if (!headers.has('User-Agent')) {
    		headers.set('User-Agent', 'node-fetch/1.0 (+https://github.com/bitinn/node-fetch)');
    	}

    	// HTTP-network-or-cache fetch step 2.15
    	if (request.compress) {
    		headers.set('Accept-Encoding', 'gzip,deflate');
    	}
    	if (!headers.has('Connection') && !request.agent) {
    		headers.set('Connection', 'close');
    	}

    	// HTTP-network fetch step 4.2
    	// chunked encoding is handled by Node.js

    	return Object.assign({}, parsedURL, {
    		method: request.method,
    		headers: exportNodeCompatibleHeaders(headers),
    		agent: request.agent
    	});
    }

    /**
     * index.js
     *
     * a request API compatible with window.fetch
     *
     * All spec algorithm step numbers are based on https://fetch.spec.whatwg.org/commit-snapshots/ae716822cb3a61843226cd090eefc6589446c1d2/.
     */

    const http = require('http');
    const https = require('https');

    var _require$3 = require('stream');

    const PassThrough$1 = _require$3.PassThrough;

    var _require2 = require('url');

    const resolve_url = _require2.resolve;

    const zlib = require('zlib');

    /**
     * Fetch function
     *
     * @param   Mixed    url   Absolute url or Request instance
     * @param   Object   opts  Fetch options
     * @return  Promise
     */
    function fetch$1(url, opts) {

    	// allow custom promise
    	if (!fetch$1.Promise) {
    		throw new Error('native promise missing, set fetch.Promise to your favorite alternative');
    	}

    	Body.Promise = fetch$1.Promise;

    	// wrap http.request into fetch
    	return new fetch$1.Promise(function (resolve, reject) {
    		// build request object
    		const request = new Request(url, opts);
    		const options = getNodeRequestOptions(request);

    		const send = (options.protocol === 'https:' ? https : http).request;

    		// send request
    		const req = send(options);
    		let reqTimeout;

    		function finalize() {
    			req.abort();
    			clearTimeout(reqTimeout);
    		}

    		if (request.timeout) {
    			req.once('socket', function (socket) {
    				reqTimeout = setTimeout(function () {
    					reject(new FetchError(`network timeout at: ${request.url}`, 'request-timeout'));
    					finalize();
    				}, request.timeout);
    			});
    		}

    		req.on('error', function (err) {
    			reject(new FetchError(`request to ${request.url} failed, reason: ${err.message}`, 'system', err));
    			finalize();
    		});

    		req.on('response', function (res) {
    			clearTimeout(reqTimeout);

    			const headers = createHeadersLenient(res.headers);

    			// HTTP fetch step 5
    			if (fetch$1.isRedirect(res.statusCode)) {
    				// HTTP fetch step 5.2
    				const location = headers.get('Location');

    				// HTTP fetch step 5.3
    				const locationURL = location === null ? null : resolve_url(request.url, location);

    				// HTTP fetch step 5.5
    				switch (request.redirect) {
    					case 'error':
    						reject(new FetchError(`redirect mode is set to error: ${request.url}`, 'no-redirect'));
    						finalize();
    						return;
    					case 'manual':
    						// node-fetch-specific step: make manual redirect a bit easier to use by setting the Location header value to the resolved URL.
    						if (locationURL !== null) {
    							headers.set('Location', locationURL);
    						}
    						break;
    					case 'follow':
    						// HTTP-redirect fetch step 2
    						if (locationURL === null) {
    							break;
    						}

    						// HTTP-redirect fetch step 5
    						if (request.counter >= request.follow) {
    							reject(new FetchError(`maximum redirect reached at: ${request.url}`, 'max-redirect'));
    							finalize();
    							return;
    						}

    						// HTTP-redirect fetch step 6 (counter increment)
    						// Create a new Request object.
    						const requestOpts = {
    							headers: new Headers(request.headers),
    							follow: request.follow,
    							counter: request.counter + 1,
    							agent: request.agent,
    							compress: request.compress,
    							method: request.method,
    							body: request.body
    						};

    						// HTTP-redirect fetch step 9
    						if (res.statusCode !== 303 && request.body && getTotalBytes(request) === null) {
    							reject(new FetchError('Cannot follow redirect with body being a readable stream', 'unsupported-redirect'));
    							finalize();
    							return;
    						}

    						// HTTP-redirect fetch step 11
    						if (res.statusCode === 303 || (res.statusCode === 301 || res.statusCode === 302) && request.method === 'POST') {
    							requestOpts.method = 'GET';
    							requestOpts.body = undefined;
    							requestOpts.headers.delete('content-length');
    						}

    						// HTTP-redirect fetch step 15
    						resolve(fetch$1(new Request(locationURL, requestOpts)));
    						finalize();
    						return;
    				}
    			}

    			// prepare response
    			let body = res.pipe(new PassThrough$1());
    			const response_options = {
    				url: request.url,
    				status: res.statusCode,
    				statusText: res.statusMessage,
    				headers: headers,
    				size: request.size,
    				timeout: request.timeout
    			};

    			// HTTP-network fetch step 12.1.1.3
    			const codings = headers.get('Content-Encoding');

    			// HTTP-network fetch step 12.1.1.4: handle content codings

    			// in following scenarios we ignore compression support
    			// 1. compression support is disabled
    			// 2. HEAD request
    			// 3. no Content-Encoding header
    			// 4. no content response (204)
    			// 5. content not modified response (304)
    			if (!request.compress || request.method === 'HEAD' || codings === null || res.statusCode === 204 || res.statusCode === 304) {
    				resolve(new Response(body, response_options));
    				return;
    			}

    			// For Node v6+
    			// Be less strict when decoding compressed responses, since sometimes
    			// servers send slightly invalid responses that are still accepted
    			// by common browsers.
    			// Always using Z_SYNC_FLUSH is what cURL does.
    			const zlibOptions = {
    				flush: zlib.Z_SYNC_FLUSH,
    				finishFlush: zlib.Z_SYNC_FLUSH
    			};

    			// for gzip
    			if (codings == 'gzip' || codings == 'x-gzip') {
    				body = body.pipe(zlib.createGunzip(zlibOptions));
    				resolve(new Response(body, response_options));
    				return;
    			}

    			// for deflate
    			if (codings == 'deflate' || codings == 'x-deflate') {
    				// handle the infamous raw deflate response from old servers
    				// a hack for old IIS and Apache servers
    				const raw = res.pipe(new PassThrough$1());
    				raw.once('data', function (chunk) {
    					// see http://stackoverflow.com/questions/37519828
    					if ((chunk[0] & 0x0F) === 0x08) {
    						body = body.pipe(zlib.createInflate());
    					} else {
    						body = body.pipe(zlib.createInflateRaw());
    					}
    					resolve(new Response(body, response_options));
    				});
    				return;
    			}

    			// otherwise, use response as-is
    			resolve(new Response(body, response_options));
    		});

    		writeToStream(req, request);
    	});
    }

    /**
     * Redirect code matching
     *
     * @param   Number   code  Status code
     * @return  Boolean
     */
    fetch$1.isRedirect = function (code) {
    	return code === 301 || code === 302 || code === 303 || code === 307 || code === 308;
    };

    // Needed for TypeScript.
    fetch$1.default = fetch$1;

    // expose Promise
    fetch$1.Promise = global.Promise;

    var utf8 = createCommonjsModule(function (module, exports) {
    (function(root) {

    	// Detect free variables `exports`
    	var freeExports = exports;

    	// Detect free variable `module`
    	var freeModule = module &&
    		module.exports == freeExports && module;

    	// Detect free variable `global`, from Node.js or Browserified code,
    	// and use it as `root`
    	var freeGlobal = typeof commonjsGlobal == 'object' && commonjsGlobal;
    	if (freeGlobal.global === freeGlobal || freeGlobal.window === freeGlobal) {
    		root = freeGlobal;
    	}

    	/*--------------------------------------------------------------------------*/

    	var stringFromCharCode = String.fromCharCode;

    	// Taken from https://mths.be/punycode
    	function ucs2decode(string) {
    		var output = [];
    		var counter = 0;
    		var length = string.length;
    		var value;
    		var extra;
    		while (counter < length) {
    			value = string.charCodeAt(counter++);
    			if (value >= 0xD800 && value <= 0xDBFF && counter < length) {
    				// high surrogate, and there is a next character
    				extra = string.charCodeAt(counter++);
    				if ((extra & 0xFC00) == 0xDC00) { // low surrogate
    					output.push(((value & 0x3FF) << 10) + (extra & 0x3FF) + 0x10000);
    				} else {
    					// unmatched surrogate; only append this code unit, in case the next
    					// code unit is the high surrogate of a surrogate pair
    					output.push(value);
    					counter--;
    				}
    			} else {
    				output.push(value);
    			}
    		}
    		return output;
    	}

    	// Taken from https://mths.be/punycode
    	function ucs2encode(array) {
    		var length = array.length;
    		var index = -1;
    		var value;
    		var output = '';
    		while (++index < length) {
    			value = array[index];
    			if (value > 0xFFFF) {
    				value -= 0x10000;
    				output += stringFromCharCode(value >>> 10 & 0x3FF | 0xD800);
    				value = 0xDC00 | value & 0x3FF;
    			}
    			output += stringFromCharCode(value);
    		}
    		return output;
    	}

    	function checkScalarValue(codePoint) {
    		if (codePoint >= 0xD800 && codePoint <= 0xDFFF) {
    			throw Error(
    				'Lone surrogate U+' + codePoint.toString(16).toUpperCase() +
    				' is not a scalar value'
    			);
    		}
    	}
    	/*--------------------------------------------------------------------------*/

    	function createByte(codePoint, shift) {
    		return stringFromCharCode(((codePoint >> shift) & 0x3F) | 0x80);
    	}

    	function encodeCodePoint(codePoint) {
    		if ((codePoint & 0xFFFFFF80) == 0) { // 1-byte sequence
    			return stringFromCharCode(codePoint);
    		}
    		var symbol = '';
    		if ((codePoint & 0xFFFFF800) == 0) { // 2-byte sequence
    			symbol = stringFromCharCode(((codePoint >> 6) & 0x1F) | 0xC0);
    		}
    		else if ((codePoint & 0xFFFF0000) == 0) { // 3-byte sequence
    			checkScalarValue(codePoint);
    			symbol = stringFromCharCode(((codePoint >> 12) & 0x0F) | 0xE0);
    			symbol += createByte(codePoint, 6);
    		}
    		else if ((codePoint & 0xFFE00000) == 0) { // 4-byte sequence
    			symbol = stringFromCharCode(((codePoint >> 18) & 0x07) | 0xF0);
    			symbol += createByte(codePoint, 12);
    			symbol += createByte(codePoint, 6);
    		}
    		symbol += stringFromCharCode((codePoint & 0x3F) | 0x80);
    		return symbol;
    	}

    	function utf8encode(string) {
    		var codePoints = ucs2decode(string);
    		var length = codePoints.length;
    		var index = -1;
    		var codePoint;
    		var byteString = '';
    		while (++index < length) {
    			codePoint = codePoints[index];
    			byteString += encodeCodePoint(codePoint);
    		}
    		return byteString;
    	}

    	/*--------------------------------------------------------------------------*/

    	function readContinuationByte() {
    		if (byteIndex >= byteCount) {
    			throw Error('Invalid byte index');
    		}

    		var continuationByte = byteArray[byteIndex] & 0xFF;
    		byteIndex++;

    		if ((continuationByte & 0xC0) == 0x80) {
    			return continuationByte & 0x3F;
    		}

    		// If we end up here, it’s not a continuation byte
    		throw Error('Invalid continuation byte');
    	}

    	function decodeSymbol() {
    		var byte1;
    		var byte2;
    		var byte3;
    		var byte4;
    		var codePoint;

    		if (byteIndex > byteCount) {
    			throw Error('Invalid byte index');
    		}

    		if (byteIndex == byteCount) {
    			return false;
    		}

    		// Read first byte
    		byte1 = byteArray[byteIndex] & 0xFF;
    		byteIndex++;

    		// 1-byte sequence (no continuation bytes)
    		if ((byte1 & 0x80) == 0) {
    			return byte1;
    		}

    		// 2-byte sequence
    		if ((byte1 & 0xE0) == 0xC0) {
    			byte2 = readContinuationByte();
    			codePoint = ((byte1 & 0x1F) << 6) | byte2;
    			if (codePoint >= 0x80) {
    				return codePoint;
    			} else {
    				throw Error('Invalid continuation byte');
    			}
    		}

    		// 3-byte sequence (may include unpaired surrogates)
    		if ((byte1 & 0xF0) == 0xE0) {
    			byte2 = readContinuationByte();
    			byte3 = readContinuationByte();
    			codePoint = ((byte1 & 0x0F) << 12) | (byte2 << 6) | byte3;
    			if (codePoint >= 0x0800) {
    				checkScalarValue(codePoint);
    				return codePoint;
    			} else {
    				throw Error('Invalid continuation byte');
    			}
    		}

    		// 4-byte sequence
    		if ((byte1 & 0xF8) == 0xF0) {
    			byte2 = readContinuationByte();
    			byte3 = readContinuationByte();
    			byte4 = readContinuationByte();
    			codePoint = ((byte1 & 0x07) << 0x12) | (byte2 << 0x0C) |
    				(byte3 << 0x06) | byte4;
    			if (codePoint >= 0x010000 && codePoint <= 0x10FFFF) {
    				return codePoint;
    			}
    		}

    		throw Error('Invalid UTF-8 detected');
    	}

    	var byteArray;
    	var byteCount;
    	var byteIndex;
    	function utf8decode(byteString) {
    		byteArray = ucs2decode(byteString);
    		byteCount = byteArray.length;
    		byteIndex = 0;
    		var codePoints = [];
    		var tmp;
    		while ((tmp = decodeSymbol()) !== false) {
    			codePoints.push(tmp);
    		}
    		return ucs2encode(codePoints);
    	}

    	/*--------------------------------------------------------------------------*/

    	var utf8 = {
    		'version': '2.1.2',
    		'encode': utf8encode,
    		'decode': utf8decode
    	};

    	// Some AMD build optimizers, like r.js, check for specific condition patterns
    	// like the following:
    	if (freeExports && !freeExports.nodeType) {
    		if (freeModule) { // in Node.js or RingoJS v0.8.0+
    			freeModule.exports = utf8;
    		} else { // in Narwhal or RingoJS v0.7.0-
    			var object = {};
    			var hasOwnProperty = object.hasOwnProperty;
    			for (var key in utf8) {
    				hasOwnProperty.call(utf8, key) && (freeExports[key] = utf8[key]);
    			}
    		}
    	} else { // in Rhino or a web browser
    		root.utf8 = utf8;
    	}

    }(commonjsGlobal));
    });
    var utf8_1 = utf8.decode;

    var StringIterator = (function (_super) {
        __extends(StringIterator, _super);
        function StringIterator() {
            return _super !== null && _super.apply(this, arguments) || this;
        }
        StringIterator.prototype.split = function (separator) {
            return new SplitIterator(this, separator);
        };
        return StringIterator;
    }(LazyIterator));
    var SplitIterator = (function (_super) {
        __extends(SplitIterator, _super);
        function SplitIterator(upstream, separator) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.impl = new SplitIteratorImpl(upstream, separator);
            return _this;
        }
        SplitIterator.prototype.summary = function () {
            return this.impl.summary();
        };
        SplitIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, this.impl.next()];
                });
            });
        };
        return SplitIterator;
    }(StringIterator));
    var SplitIteratorImpl = (function (_super) {
        __extends(SplitIteratorImpl, _super);
        function SplitIteratorImpl(upstream, separator) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.separator = separator;
            _this.carryover = '';
            return _this;
        }
        SplitIteratorImpl.prototype.summary = function () {
            return this.upstream.summary() + " -> Split('" + this.separator + "')";
        };
        SplitIteratorImpl.prototype.pump = function () {
            return __awaiter(this, void 0, void 0, function () {
                var chunkResult, lines, _i, _a, line;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0: return [4, this.upstream.next()];
                        case 1:
                            chunkResult = _b.sent();
                            if (chunkResult.done) {
                                if (this.carryover === '') {
                                    return [2, false];
                                }
                                this.outputQueue.push(this.carryover);
                                this.carryover = '';
                                return [2, true];
                            }
                            lines = chunkResult.value.split(this.separator);
                            lines[0] = this.carryover + lines[0];
                            for (_i = 0, _a = lines.slice(0, -1); _i < _a.length; _i++) {
                                line = _a[_i];
                                this.outputQueue.push(line);
                            }
                            this.carryover = lines[lines.length - 1];
                            return [2, true];
                    }
                });
            });
        };
        return SplitIteratorImpl;
    }(OneToManyIterator));

    var ByteChunkIterator = (function (_super) {
        __extends(ByteChunkIterator, _super);
        function ByteChunkIterator() {
            return _super !== null && _super.apply(this, arguments) || this;
        }
        ByteChunkIterator.prototype.decodeUTF8 = function () {
            return new Utf8Iterator(this);
        };
        return ByteChunkIterator;
    }(LazyIterator));
    var Utf8Iterator = (function (_super) {
        __extends(Utf8Iterator, _super);
        function Utf8Iterator(upstream) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.impl = new Utf8IteratorImpl(upstream);
            return _this;
        }
        Utf8Iterator.prototype.summary = function () {
            return this.impl.summary();
        };
        Utf8Iterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, this.impl.next()];
                });
            });
        };
        return Utf8Iterator;
    }(StringIterator));
    var Utf8IteratorImpl = (function (_super) {
        __extends(Utf8IteratorImpl, _super);
        function Utf8IteratorImpl(upstream) {
            var _this = _super.call(this) || this;
            _this.upstream = upstream;
            _this.partial = new Uint8Array([]);
            _this.partialBytesValid = 0;
            return _this;
        }
        Utf8IteratorImpl.prototype.summary = function () {
            return this.upstream.summary() + " -> Utf8";
        };
        Utf8IteratorImpl.prototype.pump = function () {
            return __awaiter(this, void 0, void 0, function () {
                var chunkResult, chunk, partialBytesRemaining, nextIndex, okUpToIndex, splitUtfWidth, bulk, reassembled;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4, this.upstream.next()];
                        case 1:
                            chunkResult = _a.sent();
                            if (chunkResult.done) {
                                if (this.partial.length === 0) {
                                    return [2, false];
                                }
                                chunk = new Uint8Array([]);
                            }
                            else {
                                chunk = chunkResult.value;
                            }
                            partialBytesRemaining = this.partial.length - this.partialBytesValid;
                            nextIndex = partialBytesRemaining;
                            okUpToIndex = nextIndex;
                            splitUtfWidth = 0;
                            while (nextIndex < chunk.length) {
                                okUpToIndex = nextIndex;
                                splitUtfWidth = utfWidth(chunk[nextIndex]);
                                nextIndex = okUpToIndex + splitUtfWidth;
                            }
                            if (nextIndex === chunk.length) {
                                okUpToIndex = nextIndex;
                            }
                            bulk = utf8_1(String.fromCharCode.apply(null, chunk.slice(partialBytesRemaining, okUpToIndex)));
                            if (partialBytesRemaining > 0) {
                                this.partial.set(chunk.slice(0, partialBytesRemaining), this.partialBytesValid);
                                reassembled = utf8_1(String.fromCharCode.apply(null, this.partial));
                                this.outputQueue.push(reassembled + bulk);
                            }
                            else {
                                this.outputQueue.push(bulk);
                            }
                            if (okUpToIndex === chunk.length) {
                                this.partial = new Uint8Array([]);
                                this.partialBytesValid = 0;
                            }
                            else {
                                this.partial = new Uint8Array(new ArrayBuffer(splitUtfWidth));
                                this.partial.set(chunk.slice(okUpToIndex), 0);
                                this.partialBytesValid = chunk.length - okUpToIndex;
                            }
                            return [2, true];
                    }
                });
            });
        };
        return Utf8IteratorImpl;
    }(OneToManyIterator));
    function utfWidth(firstByte) {
        if (firstByte >= 252) {
            return 6;
        }
        else if (firstByte >= 248) {
            return 5;
        }
        else if (firstByte >= 240) {
            return 4;
        }
        else if (firstByte >= 224) {
            return 3;
        }
        else if (firstByte >= 192) {
            return 2;
        }
        else {
            return 1;
        }
    }

    var FileChunkIterator = (function (_super) {
        __extends(FileChunkIterator, _super);
        function FileChunkIterator(file, options) {
            if (options === void 0) { options = {}; }
            var _this = _super.call(this) || this;
            _this.file = file;
            _this.options = options;
            util_6((file instanceof Uint8Array) ||
                (tf.ENV.get('IS_BROWSER') ?
                    (file instanceof File || file instanceof Blob) :
                    false), 'FileChunkIterator only supports File, Blob and Uint8Array right now.');
            _this.offset = options.offset || 0;
            _this.chunkSize = options.chunkSize || 1024 * 1024;
            return _this;
        }
        FileChunkIterator.prototype.summary = function () {
            return "FileChunks " + this.file;
        };
        FileChunkIterator.prototype.next = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                var chunk, _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            if (this.offset >= ((this.file instanceof Uint8Array) ?
                                this.file.byteLength :
                                this.file.size)) {
                                return [2, { value: null, done: true }];
                            }
                            chunk = new Promise(function (resolve, reject) {
                                var end = _this.offset + _this.chunkSize;
                                if (_this.file instanceof Uint8Array) {
                                    resolve(new Uint8Array(_this.file.slice(_this.offset, end)));
                                }
                                else {
                                    var fileReader_1 = new FileReader();
                                    fileReader_1.onload = function (event) {
                                        var data = fileReader_1.result;
                                        if (data instanceof ArrayBuffer) {
                                            data = new Uint8Array(data);
                                        }
                                        if (!(data instanceof Uint8Array)) {
                                            return reject(new TypeError('FileReader returned unknown type.'));
                                        }
                                        resolve(data);
                                    };
                                    fileReader_1.onabort = function (event) {
                                        return reject(new Error('Aborted'));
                                    };
                                    fileReader_1.onerror = function (event) {
                                        return reject(new Error(event.type));
                                    };
                                    var slice = _this.file.slice(_this.offset, end);
                                    fileReader_1.readAsArrayBuffer(slice);
                                }
                                _this.offset = end;
                            });
                            _a = {};
                            return [4, chunk];
                        case 1: return [2, (_a.value = (_b.sent()), _a.done = false, _a)];
                    }
                });
            });
        };
        return FileChunkIterator;
    }(ByteChunkIterator));

    function urlChunkIterator(url, options) {
        if (options === void 0) { options = {}; }
        return __awaiter(this, void 0, void 0, function () {
            var response, blob, unitArray;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!tf.ENV.get('IS_BROWSER')) return [3, 5];
                        return [4, fetch(url)];
                    case 1:
                        response = _a.sent();
                        if (!response.ok) return [3, 3];
                        return [4, response.blob()];
                    case 2:
                        blob = _a.sent();
                        return [2, new FileChunkIterator(blob, options)];
                    case 3: throw new Error(response.statusText);
                    case 4: return [3, 9];
                    case 5:
                        if (typeof url !== 'string') {
                            throw new Error('URL must be a string. Request objects are not supported ' +
                                'in the node.js environment yet.');
                        }
                        return [4, fetch$1(url)];
                    case 6:
                        response = _a.sent();
                        if (!response.ok) return [3, 8];
                        return [4, response.buffer()];
                    case 7:
                        unitArray = _a.sent();
                        return [2, new FileChunkIterator(unitArray, options)];
                    case 8: throw new Error(response.statusText);
                    case 9: return [2];
                }
            });
        });
    }

    var URLDataSource = (function (_super) {
        __extends(URLDataSource, _super);
        function URLDataSource(url, fileOptions) {
            if (fileOptions === void 0) { fileOptions = {}; }
            var _this = _super.call(this) || this;
            _this.url = url;
            _this.fileOptions = fileOptions;
            return _this;
        }
        URLDataSource.prototype.iterator = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, urlChunkIterator(this.url, this.fileOptions)];
                });
            });
        };
        return URLDataSource;
    }(DataSource));

    function csv(source, csvConfig) {
        if (csvConfig === void 0) { csvConfig = {}; }
        return new CSVDataset(new URLDataSource(source), csvConfig);
    }

    var FileDataSource = (function (_super) {
        __extends(FileDataSource, _super);
        function FileDataSource(input, options) {
            if (options === void 0) { options = {}; }
            var _this = _super.call(this) || this;
            _this.input = input;
            _this.options = options;
            return _this;
        }
        FileDataSource.prototype.iterator = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, new FileChunkIterator(this.input, this.options)];
                });
            });
        };
        return FileDataSource;
    }(DataSource));

    var version = '0.0.1';

    exports.array = array;
    exports.Dataset = Dataset;
    exports.datasetFromIteratorFn = datasetFromIteratorFn;
    exports.zip = zip;
    exports.CSVDataset = CSVDataset;
    exports.TextLineDataset = TextLineDataset;
    exports.csv = csv;
    exports.FileDataSource = FileDataSource;
    exports.URLDataSource = URLDataSource;
    exports.version_data = version;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-data.js.map
