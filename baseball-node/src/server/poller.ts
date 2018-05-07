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
import {getDatePitches} from 'baseball-pitchfx-data';
import {Subject} from 'rxjs';
import {timer} from 'rxjs/observable/timer';

export class PitchPoller {
  private _poller = new Subject();

  load = true;

  get poller() {
    return this._poller;
  }

  poll() {
    timer(500, 10000)
        .flatMap(value => {
          const date = new Date();
          if (!this.load) {
            date.setDate(date.getDate() - 1);
            this.load = true;
          }
          console.log('Getting pitches for ', date);
          return getDatePitches(date);
        })
        .startWith([])
        .pairwise()
        .map(([a, b]) => {
          return !!a ? b.slice(a.length) : b;
        })
        .subscribe((value => this.poller.next(value)));
  }
}
