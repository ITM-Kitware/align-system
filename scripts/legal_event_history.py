"""Render only observed Blue evidence with reversible compact connection records."""

import json

from prepare_blue_prompt_screen import compact_history, table_rows


RAW_GUIDE='''Additional raw evidence is supplied only in raw-event conditions.
It is the observation returned to Blue, not a transcript of Red actions.
Each raw host record preserves observed Processes and Files. Connection arrays
have columns local_address, local_port, remote_address, remote_port, extra_fields,
present_mask. Bits 1,2,4,8 say which of the first four keys were present.
An address written @Hostname abbreviates its IP in the visible table; other
addresses remain literal. This permits observing connection direction and ports.
Local/remote do not automatically imply attacker/victim or successful exploitation.
Process metadata and additional connection fields remain literal observed values.
Host attributes report the keys present in that observation and changes from the
last observed values. A missing attribute is unobserved, not evidence of absence.
Every process and connection occurrence is retained in its observed order.
There is no supplied Red action success flag. Blue action results remain explicit.
'''


def raw_history(capture,start,end):
    aliases={row[1]:'@'+host for host,row in table_rows(capture['steps'][start]['defender_observation']).items()}
    attributes={}
    timeline=[]
    for index in range(start,end):
        observed=capture['steps'][index]['defender_raw_observation']
        hosts=[]
        for host,values in observed.items():
            if host=='success':continue
            if not isinstance(values,dict):raise ValueError('Unexpected raw host observation')
            if set(values)-{'Processes','Files','Interface','System info'}:
                raise ValueError('Raw observation has an unreviewed field')
            record={'host':host}
            keys=[k for k in values if k not in ('Processes','Files')]
            previous=attributes.setdefault(host,{})
            updates={k:values[k] for k in keys if k not in previous or previous[k]!=values[k]}
            record['attributes_present']=keys
            record['attribute_updates']=updates
            previous.update(updates)
            if 'Processes' in values:
                processes=[]
                for process in values['Processes']:
                    metadata={k:v for k,v in process.items() if k!='Connections'}
                    item={'fields':metadata}
                    if 'Connections' in process:
                        connections=[]
                        for connection in process['Connections']:
                            names=('local_address','local_port','remote_address','remote_port')
                            row=[connection.get(k) for k in names]
                            for address_index in (0,2):
                                row[address_index]=aliases.get(row[address_index],row[address_index])
                            extra={k:v for k,v in connection.items() if k not in names}
                            mask=sum(1<<i for i,k in enumerate(names) if k in connection)
                            connections.append([*row,extra,mask])
                        item['connections']=connections
                    processes.append(item)
                record['processes']=processes
            if 'Files' in values:record['files']=values['Files']
            hosts.append(record)
        timeline.append([index,hosts])
    return json.dumps(timeline,separators=(',',':'),sort_keys=True)


def event_history(capture,start,end,raw):
    result=compact_history(capture,start,end)
    if raw:
        result+='\nAdditional raw Blue observations [step, host_records]:\n'+raw_history(capture,start,end)
    return result
