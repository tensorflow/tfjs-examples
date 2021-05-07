import { Injectable } from '@angular/core';

import { Merch } from './merch';

@Injectable({
  providedIn: 'root',
})
export class MerchService {
  getMerchList(category: string): Promise<Merch[]> {
    return fetch(
      'http://localhost:8000/api/merch/' + ((category || '').trim() || 'all')
    ).then((response) => response.json());
  }
}
